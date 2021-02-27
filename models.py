# models
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as utils
import scipy.stats as stats
import json

from torch.distributions.beta import Beta
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

class TabularFeatNet(nn.Module):
    def __init__(self):
        super(TabularFeatNet, self).__init__()
        self.ops = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU())
    def forward(self, x):
        return self.ops(x)
    
def conv5x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)

def conv9x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4)

def conv15x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=stride, padding=7)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm1d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv15x1(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv15x1(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
#             print("in if")
#             print(x.shape)
            identity = self.downsample(x)
        
#         print("In block forward")
#         print(self.downsample)
#         print(x.shape)
#         print(identity.shape)
#         print(out.shape)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_outputs=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=nn.BatchNorm1d):

        super(ResNet, self).__init__()
        
        self.num_outputs = num_outputs
        
        self.tabfeatnet = TabularFeatNet()

        self._norm_layer = norm_layer

        self.inplanes = 32

        self.conv1 = nn.Conv1d(12, self.inplanes, kernel_size=15, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
#         self.layer5 = self._make_layer(block, 256, layers[4], stride=2)
#         self.layer6 = self._make_layer(block, 256, layers[5], stride=2)
#         self.layer7 = self._make_layer(block, 256, layers[6], stride=1)
#         self.layer8 = self._make_layer(block, 256, layers[7], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256+128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))              

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
#             print("Got to downsample place")
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, x_tab):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = self.layer7(x)
#         x = self.layer8(x)

#         print("bef avgpool", x.shape)
        x = self.avgpool(x)
#         print("bef flat", x.shape)
        x = torch.flatten(x, 1)
#         print("post flat", x.shape)
        
        x_tab = self.tabfeatnet(x_tab)
    
        # concat
        x = torch.cat([x, x_tab], dim=1)
        x = self.fc(x)
        return x

    def forward(self, x, x_other):
        BS, L, C = x.shape
        x = x.transpose(1,2)
        op = self._forward_impl(x, x_other)
        op_pi = torch.sigmoid(op)
        return op_pi


def resnet(arch, block, layers, pretrained2, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model




class ResNetVI(nn.Module):
    def __init__(self, block, layers, num_outputs=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=nn.BatchNorm1d):

        super(ResNetVI, self).__init__()
        
        self.num_outputs = num_outputs
    
        self.tabfeatnet = TabularFeatNet()

        self._norm_layer = norm_layer

        self.inplanes = 32

        self.conv1 = nn.Conv1d(12, self.inplanes, kernel_size=15, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256+128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs*2))     

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
#             print("Got to downsample place")
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, x_tab):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x_tab = self.tabfeatnet(x_tab)
    
        # concat
        x = torch.cat([x, x_tab], dim=1)
        x = self.fc(x)
        return x

    def forward(self, x, x_other):
        BS, L, C = x.shape
        x = x.transpose(1,2)
        op = self._forward_impl(x, x_other)
        
        op_z = op[:, :(self.num_outputs-1)*2]
        op_pi = op[:, (self.num_outputs-1)*2:]
        
        op_z_mean = op_z[:, :(self.num_outputs-1)]
        op_z_std = torch.exp(op_z[:, (self.num_outputs-1):])

        op_pi_alpha = 1.0 + 10.0* torch.sigmoid(op_pi[:, 0])
        op_pi_beta = 1.0 + 10.0* torch.sigmoid(op_pi[:, 1])

        return op_z_mean, op_z_std, op_pi_alpha, op_pi_beta


def resnet_vi(arch, block, layers, pretrained2, progress, **kwargs):
    model = ResNetVI(block, layers, **kwargs)
    return model


class ResNetInf(nn.Module):

    def __init__(self, block, layers, num_outputs=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=nn.BatchNorm1d):

        super(ResNetInf, self).__init__()
        
        self.num_outputs = num_outputs
        
        self.tabfeatnet = TabularFeatNet()

        self._norm_layer = norm_layer

        self.inplanes = 32

        self.conv1 = nn.Conv1d(12, self.inplanes, kernel_size=15, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256+128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs))     

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
#             print("Got to downsample place")
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, x_tab):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x_tab = self.tabfeatnet(x_tab)
    
        # concat
        x = torch.cat([x, x_tab], dim=1)
        x = self.fc(x)
        return x

    def forward(self, x, x_other):
        BS, L, C = x.shape
        x = x.transpose(1,2)
        op = self._forward_impl(x, x_other)
        
        op_z = op[:, :-1]
        op_pi = op[:, -1]
        
        op_z = torch.exp(op_z)
        op_pi = torch.sigmoid(op_pi)

        return op_z, op_pi


def resnet_inf(arch, block, layers, pretrained2, progress, **kwargs):
    model = ResNetInf(block, layers, **kwargs)
    return model



class DecoderECG(nn.Module):
    def __init__(self, num_outputs):
        super(DecoderECG, self).__init__()
        self.num_outputs = num_outputs
        
        self.fc1= nn.Sequential(
            nn.Linear(num_outputs, 100),
            nn.LeakyReLU()
        )
        
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=1.5),
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        # op has len150
        
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=1),
            nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        # op has len 150
        
        self.layer4 = nn.Sequential(
            nn.Upsample(scale_factor=1.5),
            nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        # op has len 225
        

        self.layer5 = nn.Sequential(
            nn.Upsample(scale_factor=1),
            nn.Conv1d(128, 128, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        # op has len 225
        
        self.layer6 = nn.Sequential(
            nn.Upsample(scale_factor=300/225),
            nn.Conv1d(128, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # op has len 300
        
        self.layer7 = nn.Sequential(
            nn.Upsample(scale_factor=4/3),
            nn.Conv1d(64,32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        # op has len 400
        
        self.layer8 = nn.Sequential(
            nn.Upsample(scale_factor=5/4),
            nn.Conv1d(32, 1, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU()
        )
        

    def forward(self, zin):
        out = self.fc1(zin)
        out = out.unsqueeze(dim=1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.squeeze()
        return out

class DecoderDemo(nn.Module):
    def __init__(self, num_outputs):
        super(DecoderDemo, self).__init__()
        
        self.fc1= nn.Sequential(
            nn.Linear(num_outputs, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(64, 7),
        )

    def forward(self, zin):
        out = self.fc1(zin)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.squeeze()
        return out