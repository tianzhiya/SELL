import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

from Seg.LoadSegFeature import LoadSegFeature
from models.archs.transformer.Models import Encoder_patch66


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class SemanticLearner(nn.Module):
    def __init__(self, input_channels=3, output_channels=64):
        super(SemanticLearner, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 3x400x608 -> 64x200x304
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x200x304 -> 128x100x152
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        self.conv1x1 = nn.Conv2d(128, output_channels, kernel_size=1)

    def forward(self, img):
        x = self.downsampling(img)
        x = self.residual_blocks(x)
        x = self.conv1x1(x)
        return x


class SemanticEmbeddModule(nn.Module):

    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = self.bn(normalized * (1 + gamma)) + beta
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeezed = F.adaptive_avg_pool2d(x, (1, 1))
        squeezed = squeezed.view(batch_size, channels)
        excitation = F.relu(self.fc1(squeezed))
        excitation = self.sigmoid(self.fc2(excitation))
        excitation = excitation.view(batch_size, channels, 1, 1)
        return x * excitation


class SemanticEmbeddModuleWithSE(nn.Module):
    def __init__(self, norm_nc, label_nc, nhidden=64, se_reduction=16):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.se = SELayer(nhidden, reduction=se_reduction)

        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        actv = self.se(actv)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = self.bn(normalized * (1 + gamma)) + beta
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeezed = F.adaptive_avg_pool2d(x, (1, 1))
        squeezed = squeezed.view(batch_size, channels)

        excitation = F.relu(self.fc1(squeezed))
        excitation = self.sigmoid(self.fc2(excitation))
        excitation = excitation.view(batch_size, channels, 1, 1)
        return x * excitation


class SemanticEmbeddModuleWithSE(nn.Module):
    def __init__(self, norm_nc, label_nc, nhidden=64, se_reduction=16):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.se = SELayer(nhidden, reduction=se_reduction)

        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        actv = self.se(actv)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = self.bn(normalized * (1 + gamma)) + beta

        return out


class low_light_transformer(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_transformer, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)
        self.conv1x1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.semanticLearner = SemanticLearner(input_channels=3, output_channels=64)
        self.SemanticEmbeddModule1 = SemanticEmbeddModuleWithSE(norm_nc=64, label_nc=64, nhidden=64)
        self.SemanticEmbeddModule2 = SemanticEmbeddModuleWithSE(norm_nc=128, label_nc=64, nhidden=64)
        self.SemanticEmbeddModule3 = SemanticEmbeddModuleWithSE(norm_nc=128, label_nc=64, nhidden=64)

    def forward(self, x, realX):

        floadSegF = LoadSegFeature(realX)
        segPrioriK = floadSegF.getVisFeature2()
        PrioriGT = self.conv1x1(segPrioriK)
        x_center = x
        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))
        fea = self.feature_extraction(L1_fea_3)
        preProir = self.semanticLearner(x)
        out_noise = self.recon_trunk(fea)
        out_noise = self.SemanticEmbeddModule1(out_noise, preProir)
        out_noise = torch.cat([L1_fea_3, out_noise], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)

        out_noise = self.SemanticEmbeddModule2(out_noise, preProir)

        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)

        out_noise = self.SemanticEmbeddModule3(out_noise, preProir)

        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x_center

        return out_noise, preProir, PrioriGT
