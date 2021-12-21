from __future__ import division, absolute_import
from functools import partial
from pickle import TRUE
from warnings import simplefilter
from caffe2.python.workspace import FetchBlob
from numpy.core.fromnumeric import shape
from numpy.core.records import format_parser
from numpy.lib.function_base import _flip_dispatcher, append
import torch
from torch.nn.modules.activation import ReLU
from torch.serialization import PROTOCOL_VERSION
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import functional as F
from .resnet import resnet50_fc512, resnet50backbone

__all__ = ['resnet3modal']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)






class Resnet3modal(nn.Module):
    def __init__(
        self,
        num_classes,
        loss,
        block,
        parts=6,
        reduced_dim=512,
        cls_dim=128,
        nonlinear='relu',
        **kwargs
    ):
        super(Resnet3modal, self).__init__()
        self.modal_number = 3
        self.loss = loss
        self.num_classes = num_classes

        # base network
        self.backbone = nn.ModuleList(
            [
                resnet50_fc512(num_classes, pretrained=True)
                for _ in range(self.modal_number)
            ]
        )


        # identity classification layer
        self.classifier_all = nn.Linear(512*self.modal_number, num_classes)
        self.classifier_RGB = nn.Linear(512, num_classes)
        self.classifier_TI = nn.Linear(512, num_classes)
        self.classifier_NI = nn.Linear(512, num_classes)

    def _fc_layer(self, in_channel, out_channel):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x, return_featuremaps=False):
        # === ResNet50 layers ===

        fc_RGB = self.backbone[0](x[0])
        fc_NI = self.backbone[1](x[1])
        fc_TI = self.backbone[2](x[2])
        # print('resnet output: ', fc_RGB.shape, fc_NI.shape, fc_TI.shape)

        fc_all = torch.cat([fc_RGB, fc_NI, fc_TI], dim=1)
        # print('all output: ', fc_all.shape)


        # classifier layers
        result = []
        result.append(self.classifier_RGB(fc_RGB))
        result.append(self.classifier_NI(fc_NI))
        result.append(self.classifier_TI(fc_TI))
        result.append(self.classifier_all(fc_all))

        # print('result: ', len(result))

        if not self.training:
            return fc_all
        elif self.loss == 'softmax':
            return result
        # elif self.loss == 'margin':
        #     return result, self.l2_norm(fc_RGB), self.l2_norm(fc_NI_all), self.l2_norm(fc_TI_all)
        # elif self.loss == 'hcloss':
        #     return result, self.l2_norm(fc_RGB_), self.l2_norm(fc_NI_all), self.l2_norm(fc_TI_all)
        # elif self.loss == 'CMT':
        #     return result, self.l2_norm(fc_RGB_all), self.l2_norm(fc_NI_all), self.l2_norm(fc_TI_all)


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def resnet3modal(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = Resnet3modal(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=6,
        reduced_dim=768,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
