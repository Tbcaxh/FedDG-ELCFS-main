# -*- coding: utf-8 -*-
"""
2D Unet-like architecture code in Pytorch
"""
import math
import numpy as np
from networks.layers import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

class MyUpsample2(nn.Module):
    def forward(self, x):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

#### Note: All are functional units except the norms, which are sequential
class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

    def forward(self, x, weights=None, layer_idx=None):

        if weights == None:
            weight_1, bias_1 = self.conv1.weight, self.conv1.bias
            weight_2, bias_2 = self.conv2.weight, self.conv2.bias
            weight_3, bias_3 = self.conv3.weight, self.conv3.bias

        else:
            weight_1, bias_1 = weights[layer_idx+'.conv1.weight'], weights[layer_idx+'.conv1.bias']
            weight_2, bias_2 = weights[layer_idx+'.conv2.weight'], weights[layer_idx+'.conv2.bias']
            weight_3, bias_3 = weights[layer_idx+'.conv3.weight'], weights[layer_idx+'.conv3.bias']

        if not self.first:
            x = maxpool2D(x, kernel_size=2)

        #layer 1 conv, bn
        x = conv2d(x, weight_1, bias_1)
        x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = conv2d(x, weight_2, bias_2)
        y = self.bn2(y)
        y = relu(y)

        #layer 3 conv, bn
        z = conv2d(y, weight_3, bias_3)
        z = self.bn3(z)
        z = relu(z)

        return z

class ConvU(nn.Module):
    def __init__(self, planes, norm='gn', first=False):
        super(ConvU, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1   = normalization(planes, norm)

        self.pool = MyUpsample2()
        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev, weights=None, layer_idx=None):

        if weights == None:
            if not self.first:
                weight_1, bias_1 = self.conv1.weight, self.conv1.bias
            weight_2, bias_2 = self.conv2.weight, self.conv2.bias
            weight_3, bias_3 = self.conv3.weight, self.conv3.bias

        else:
            if not self.first:
                weight_1, bias_1 = weights[layer_idx+'.conv1.weight'], weights[layer_idx+'.conv1.bias']
            weight_2, bias_2 = weights[layer_idx+'.conv2.weight'], weights[layer_idx+'.conv2.bias']
            weight_3, bias_3 = weights[layer_idx+'.conv3.weight'], weights[layer_idx+'.conv3.bias']
            
        #layer 1 conv, bn, relu
        if not self.first:
            x = conv2d(x, weight_1, bias_1, )
            x = self.bn1(x)
            x = relu(x)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = conv2d(y, weight_2, bias_2, kernel_size=1, stride=1, padding=0)
        y = self.bn2(y)
        y = relu(y)

        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = conv2d(y, weight_3, bias_3)
        y = self.bn3(y)
        y = relu(y)

        return y


class ResidualAdapter2D(nn.Module):
    def __init__(self, channels, bottleneck=8, scale=1.0):
        super(ResidualAdapter2D, self).__init__()
        bottleneck = max(1, int(bottleneck))
        self.down = nn.Conv2d(channels, bottleneck, 1, 1, 0, bias=True)
        self.up = nn.Conv2d(bottleneck, channels, 1, 1, 0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.scale = float(scale)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.down.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.down.bias, 0)
        nn.init.constant_(self.up.weight, 0)
        nn.init.constant_(self.up.bias, 0)

    def forward(self, x, weights=None, layer_idx='adapter'):
        if weights is None:
            residual = self.down(x)
            residual = self.relu(residual)
            residual = self.up(residual)
        else:
            residual = conv2d(
                x,
                weights[layer_idx + '.down.weight'],
                weights[layer_idx + '.down.bias'],
                kernel_size=1,
                stride=1,
                padding=0
            )
            residual = relu(residual)
            residual = conv2d(
                residual,
                weights[layer_idx + '.up.weight'],
                weights[layer_idx + '.up.bias'],
                kernel_size=1,
                stride=1,
                padding=0
            )
        return x + residual * self.scale


class SpatialGatedAdapter2D(nn.Module):
    def __init__(self, channels, bottleneck=8, scale=1.0, gate_channels=8, gate_bias=-3.0):
        super(SpatialGatedAdapter2D, self).__init__()
        bottleneck = max(1, int(bottleneck))
        gate_channels = max(1, int(gate_channels))
        self.down = nn.Conv2d(channels, bottleneck, 1, 1, 0, bias=True)
        self.up = nn.Conv2d(bottleneck, channels, 1, 1, 0, bias=True)
        self.gate_conv1 = nn.Conv2d(channels, gate_channels, 3, 1, 1, bias=False)
        self.gate_norm = nn.GroupNorm(1, gate_channels)
        self.gate_conv2 = nn.Conv2d(gate_channels, 1, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.scale = float(scale)
        self.gate_bias = float(gate_bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.down.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.down.bias, 0)
        nn.init.constant_(self.up.weight, 0)
        nn.init.constant_(self.up.bias, 0)
        nn.init.kaiming_normal_(self.gate_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.gate_norm.weight, 1)
        nn.init.constant_(self.gate_norm.bias, 0)
        nn.init.kaiming_normal_(self.gate_conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.gate_conv2.bias, self.gate_bias)

    def forward(self, x, weights=None, layer_idx='adapter'):
        if weights is None:
            residual = self.down(x)
            residual = self.relu(residual)
            residual = self.up(residual)

            gate = self.gate_conv1(x)
            gate = self.gate_norm(gate)
            gate = self.relu(gate)
            gate = self.gate_conv2(gate)
        else:
            residual = conv2d(
                x,
                weights[layer_idx + '.down.weight'],
                weights[layer_idx + '.down.bias'],
                kernel_size=1,
                stride=1,
                padding=0
            )
            residual = relu(residual)
            residual = conv2d(
                residual,
                weights[layer_idx + '.up.weight'],
                weights[layer_idx + '.up.bias'],
                kernel_size=1,
                stride=1,
                padding=0
            )

            gate = conv2d(
                x,
                weights[layer_idx + '.gate_conv1.weight'],
                weights.get(layer_idx + '.gate_conv1.bias', None),
                kernel_size=3,
                stride=1,
                padding=1
            )
            gate = F.group_norm(
                gate,
                self.gate_norm.num_groups,
                weight=weights[layer_idx + '.gate_norm.weight'],
                bias=weights[layer_idx + '.gate_norm.bias']
            )
            gate = relu(gate)
            gate = conv2d(
                gate,
                weights[layer_idx + '.gate_conv2.weight'],
                weights[layer_idx + '.gate_conv2.bias'],
                kernel_size=3,
                stride=1,
                padding=1
            )
        gate = torch.sigmoid(gate)
        return x + residual * gate * self.scale


class Unet2D(nn.Module):
    def __init__(self, c=3, n=16, norm='in', num_classes=2,
                 use_adapter=True, adapter_bottleneck=8, adapter_scale=1.0,
                 adapter_type='residual', gate_bias=-3.0):
        super(Unet2D, self).__init__()
        self.use_adapter = bool(use_adapter)
        self.adapter_type = adapter_type

        self.convd1 = ConvD(c,     n, norm, first=True)
        self.convd2 = ConvD(n,   2*n, norm)
        self.convd3 = ConvD(2*n, 4*n, norm)
        self.convd4 = ConvD(4*n, 8*n, norm)
        self.convd5 = ConvD(8*n,16*n, norm)

        self.convu4 = ConvU(16*n, norm, first=True)
        self.convu3 = ConvU(8*n, norm)
        self.convu2 = ConvU(4*n, norm)
        self.convu1 = ConvU(2*n, norm)

        self.seg1 = nn.Conv2d(2*n, num_classes, 1)
        self.adapter = None
        if self.use_adapter:
            if adapter_type == 'residual':
                self.adapter = ResidualAdapter2D(2*n, adapter_bottleneck, adapter_scale)
            elif adapter_type in ('spatial_gate', 'spatial_gated'):
                self.adapter = SpatialGatedAdapter2D(2*n, adapter_bottleneck, adapter_scale,
                                                         gate_bias=gate_bias)
            else:
                raise ValueError('adapter_type {} is not supported'.format(adapter_type))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.adapter is not None:
            self.adapter.reset_parameters()

    def forward(self, x, weights=None, use_adapter=True):
        # stop_gradient = Tru
        if weights == None:
            x1 = self.convd1(x)
            x2 = self.convd2(x1)
            x3 = self.convd3(x2)
            x4 = self.convd4(x3)
            x5 = self.convd5(x4)

            y4 = self.convu4(x5, x4)
            y3 = self.convu3(y4, x3)
            y2 = self.convu2(y3, x2)
            y1 = self.convu1(y2, x1)

            embedding = y1
            if self.adapter is not None and use_adapter:
                y1 = self.adapter(y1)


            y1_pred = conv2d(y1, self.seg1.weight, self.seg1.bias, kernel_size=None, stride=1, padding=0) #+ upsample(y2)
            # print (y1.shape)
        else:
            x1 = self.convd1(x, weights=weights, layer_idx='convd1')
            x2 = self.convd2(x1, weights=weights, layer_idx='convd2')
            x3 = self.convd3(x2, weights=weights, layer_idx='convd3')
            x4 = self.convd4(x3, weights=weights, layer_idx='convd4')
            x5 = self.convd5(x4, weights=weights, layer_idx='convd5')
            y4 = self.convu4(x5, x4, weights=weights, layer_idx='convu4')
            y3 = self.convu3(y4, x3, weights=weights, layer_idx='convu3')
            y2 = self.convu2(y3, x2, weights=weights, layer_idx='convu2')
            y1 = self.convu1(y2, x1, weights=weights, layer_idx='convu1')

            embedding = y1
            if self.adapter is not None and use_adapter:
                y1 = self.adapter(y1, weights=weights, layer_idx='adapter')

            y1_pred = conv2d(y1, weights['seg1.weight'], weights['seg1.bias'], kernel_size=None, stride=1, padding=0) #+ upsample(y2)
        predictions = torch.sigmoid(input=y1_pred)

        # 返回 adapter 前的 decoder 特征，便于内容原型和路由查询保持同一特征空间。
        return y1_pred, predictions, embedding
