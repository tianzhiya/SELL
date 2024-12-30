import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from datasets.LOLv2.Real_captured.Args import options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.',
                        default='./datasets/LOLv2/Real_captured/Args/Train/LOLv2_real.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    rank = distribSet(args, opt)
    resume_state = loadSate(opt)
    opt = option.dict_to_nonedict(opt)
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        print('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    total_epochs, total_iters, train_loader, train_sampler = dataLoader(opt, rank)
    model = create_model(opt)

    if resume_state:
        print('Resuming training from epoch: {}, iter: {}.'.format(resume_state['epoch'], resume_state['iter']))
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)
        del resume_state
    else:
        current_step = 0
        start_epoch = 0

    train(current_step, model, opt, rank, start_epoch, total_epochs, total_iters, train_loader,
          train_sampler)


def distribSet(args, opt):
    if args.launcher == 'none':
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    return rank


def dataLoader(opt, rank):
    torch.backends.cudnn.benchmark = True
    dataset_ratio = 200
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
                print('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                print('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    return total_epochs, total_iters, train_loader, train_sampler


def mkDirLog(opt, rank):
    if rank <= 0:
        logger = logging.getLogger('base')
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:
                from torch.utils.tensorboard import SummaryWriter
            else:
                print(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='./tb_logger/' + opt['name'])
    else:
        logger = logging.getLogger('base')
    return logger, tb_logger


def train(current_step, model, opt, rank, start_epoch, total_epochs, total_iters, train_loader,
          train_sampler):
    print('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)

                if rank <= 0:
                    print(message)
    if rank <= 0:
        print('Saving the final model.')
        model.save('SELL')
        print('End of training.')


def loadSate(opt):
    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])
    else:
        resume_state = None
    return resume_state


if __name__ == '__main__':
    main()
