import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, VGGLoss, ContentLoss

logger = logging.getLogger('base')


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
        train_opt = opt['train']

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            self.cri_pix_ill = nn.L1Loss(reduction='sum').to(self.device)

            self.cri_vgg = VGGLoss()
            self.cri_content = ContentLoss()

            self.priorLoss = nn.L1Loss(reduction='none')

            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.nf = data['nf'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()

        self.fake_H, preProir, PrioriGT = self.netG(self.var_L, self.real_H)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_vgg = self.l_pix_w * self.cri_vgg(self.fake_H, self.real_H) * 0.1
        l_content = self.cri_content(self.fake_H, self.real_H)

        l_final = l_pix + l_vgg + l_content + 0.1 * self.priorLoss(preProir, PrioriGT).mean()
        l_final.backward()
        self.optimizer_G.step()
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_vgg'] = l_vgg.item()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()

        dark = self.var_L
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        if not (len(self.nf.shape) == 4):
            self.nf = self.nf.unsqueeze(dim=0)
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        mask = mask.repeat(1, 3, 1, 1)
        out_dict['rlt3'] = mask[0].float().cpu()

        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        out_dict['ill'] = mask[0].float().cpu()
        out_dict['rlt2'] = self.nf.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del dark
        del light
        del mask
        del noise
        del self.real_H
        del self.nf
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'Final', iter_label)
