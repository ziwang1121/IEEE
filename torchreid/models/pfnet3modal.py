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
from .resnet import resnet50_ieee, resnet50backbone

__all__ = ['pfnet3modal']

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




class normalize(nn.Module):
    def __init__(self, power=2):
        super(normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



class PFNET3modalPart(nn.Module):
    def __init__(
        self,
        num_classes,
        loss,
        block,
        parts=1,
        reduced_dim=512,
        cls_dim=128,
        nonlinear='relu',
        **kwargs
    ):
        modal_number = 3
        fc_dims = [256]
        fc_dims_fuse = [256]
        super(PFNET3modalPart, self).__init__()
        self.loss = loss
        self.parts = 2
        
        self.backbone = nn.ModuleList(
            [
                resnet50_ieee(num_classes, pretrained=True)
                for _ in range(modal_number)
            ]
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))

        self.fc_R = nn.ModuleList(
            [   
                self._construct_fc_layer(fc_dims, 2048, None)
                for _ in range(self.parts)
            ]
        )
        self.fc_T = nn.ModuleList(
            [   
                self._construct_fc_layer(fc_dims, 2048, None)
                for _ in range(self.parts)
            ]
        )
        self.fc_N = nn.ModuleList(
            [   
                self._construct_fc_layer(fc_dims, 2048, None)
                for _ in range(self.parts)
            ]
        )
        self.fc_RT = nn.ModuleList(
            [   
                self._construct_fc_layer(fc_dims_fuse, 2048, None)
                for _ in range(self.parts)
            ]
        )
        self.fc_RN = nn.ModuleList(
            [   
                self._construct_fc_layer(fc_dims_fuse, 2048, None)
                for _ in range(self.parts)
            ]
        )

        self.classifier_all = nn.Linear(fc_dims[0]*5*self.parts, num_classes)
        self.classifier_R = nn.ModuleList(
            [   
                nn.Linear(fc_dims[0], num_classes)
                for _ in range(self.parts)
            ]
        )
        self.classifier_N = nn.ModuleList(
            [   
                nn.Linear(fc_dims[0], num_classes)
                for _ in range(self.parts)
            ]
        )
        self.classifier_T = nn.ModuleList(
            [   
                nn.Linear(fc_dims[0], num_classes)
                for _ in range(self.parts)
            ]
        )
        self.classifier_RT = nn.ModuleList(
            [   
                nn.Linear(fc_dims[0], num_classes)
                for _ in range(self.parts)
            ]
        )
        self.classifier_RN = nn.ModuleList(
            [   
                nn.Linear(fc_dims[0], num_classes)
                for _ in range(self.parts)
            ]
        )

        self.l2_norm = normalize()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer
        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)


    def forward(self, x, return_featuremaps=False):
        resnet_R = self.backbone[0](x[0])
        resnet_N = self.backbone[1](x[1])
        resnet_T = self.backbone[2](x[2])

        resnet_RT = resnet_R + resnet_T
        resnet_RN = resnet_R + resnet_N


        pooling_R = self.global_avgpool(resnet_R)
        pooling_N = self.global_avgpool(resnet_N)
        pooling_T = self.global_avgpool(resnet_T)
        pooling_RT = self.global_avgpool(resnet_RT)
        pooling_RN = self.global_avgpool(resnet_RN)
        
        part_R = []; part_N = []; part_T = []; part_RT = []; part_RN = []
        for i in range(self.parts):
            part_R.append((pooling_R[:, :, i, :]).view((pooling_R[:, :, i, :]).size(0), -1))
            part_N.append((pooling_N[:, :, i, :]).view((pooling_N[:, :, i, :]).size(0), -1))
            part_T.append((pooling_T[:, :, i, :]).view((pooling_T[:, :, i, :]).size(0), -1))   
            part_RT.append((pooling_RT[:, :, i, :]).view((pooling_RT[:, :, i, :]).size(0), -1))   
            part_RN.append((pooling_RN[:, :, i, :]).view((pooling_RN[:, :, i, :]).size(0), -1))   


        fc_R = []; fc_N = []; fc_T = []; fc_RT = []; fc_RN = []
        for i in range(self.parts):
            fc_R.append(self.fc_R[i](part_R[i]))
            fc_N.append(self.fc_N[i](part_N[i]))
            fc_T.append(self.fc_T[i](part_T[i]))
            fc_RT.append(self.fc_RT[i](part_RT[i]))
            fc_RN.append(self.fc_RN[i](part_RN[i]))

        fc_R_all = torch.cat(fc_R, dim=1)
        fc_N_all = torch.cat(fc_N, dim=1)
        fc_T_all = torch.cat(fc_T, dim=1)
        fc_RT_all = torch.cat(fc_RT, dim=1)
        fc_RN_all = torch.cat(fc_RN, dim=1)
        fc_all = torch.cat([fc_T_all, fc_RT_all, fc_R_all, fc_RN_all, fc_N_all], dim=1)
        
        # print(fc_all.shape)

        if not self.training:
            return fc_all
        
        result = []
        for i in range(self.parts):
            result.append(self.classifier_R[i](fc_R[i]))
            result.append(self.classifier_N[i](fc_N[i]))
            result.append(self.classifier_T[i](fc_T[i]))
            result.append(self.classifier_RT[i](fc_RT[i]))
            result.append(self.classifier_RN[i](fc_RN[i]))
        # result.append(self.classifier_all(fc_all))


        if self.loss == 'softmax':
            return result
        elif self.loss == 'triplet':
            return result, fc_all
        elif self.loss == 'hcloss':
            return result, self.l2_norm(fc_R), self.l2_norm(fc_N), self.l2_norm(fc_T)
        elif self.loss == 'CMT':
            return result, self.l2_norm(fc_R), self.l2_norm(fc_N), self.l2_norm(fc_T)


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


def pfnet3modal(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PFNET3modalPart(
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
