from __future__ import division, absolute_import
from functools import partial
from pickle import TRUE
from warnings import simplefilter
# # from caffe2.python.workspace import FetchBlob
from numpy.core.fromnumeric import shape
from numpy.core.records import format_parser
from numpy.lib.function_base import _flip_dispatcher, append
import torch
from torch.nn.modules.activation import ReLU
from torch.serialization import PROTOCOL_VERSION
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import functional as F
from .resnet import resnet50backbone

__all__ = ['pfnet']

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


class FeaturemapGate(nn.Module):
    def __init__(self, in_planes):
        super(FeaturemapGate, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv = DimReduceLayer(in_planes, 1, 'relu')

    def forward(self, x):
        x = self.GAP(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class normalize(nn.Module):
    def __init__(self, power=2):
        super(normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



class crossAttention(nn.Module):
    def __init__(self, in_dim):
        super(crossAttention, self).__init__()

        self.in_channels = in_dim
        self.inter_channels = in_dim // 2

        # function g in the paper which goes through conv. with kernel size 1
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        self.W_z = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                nn.BatchNorm2d(self.in_channels)
            )
        # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

        # define theta and phi for all operations except gaussian
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
    
    def forward(self, f_part, f_global):
        batch_size = f_global.size(0)
        
        # (N, C, THW)
        # vs
        g_x = self.g(f_global).view(batch_size, self.inter_channels, -1)
        # g_x = self.g(f_global)
        # print(g_x.shape)
        g_x = g_x.permute(0, 2, 1)
    
        
        # qt
        theta_x = self.theta(f_part).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # ks
        phi_x = self.phi(f_global).view(batch_size, self.inter_channels, -1)
        
        # similarity
        f = torch.matmul(theta_x, phi_x)

        # softmax
        f_div_C = F.softmax(f, dim=-1)
        
        # new vs
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *f_global.size()[2:])
        
        W_y = self.W_z(y)
        
        # residual connection
        z = W_y + f_part

        return z



class selfAttention(nn.Module):
    def __init__(self, in_dim):
        super(selfAttention, self).__init__()

        self.in_channels = in_dim
        self.inter_channels = in_dim // 2

        # function g in the paper which goes through conv. with kernel size 1
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        self.W_z = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                nn.BatchNorm2d(self.in_channels)
            )
        # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

        # define theta and phi for all operations except gaussian
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
    
    def forward(self, avgRest):
        batch_size = avgRest.size(0)
        
        # (N, C, THW)
        g_x = self.g(avgRest).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(avgRest).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(avgRest).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)

        # softmax
        f_div_C = F.softmax(f, dim=-1)
        

        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *avgRest.size()[2:])
        
        W_y = self.W_z(y)
        
        # residual connection
        z = W_y + avgRest

        return z


class coarseGrainedFusion(nn.Module):
    def __init__(self, in_planes):
        reduce_planes = in_planes // 2
        super(coarseGrainedFusion, self).__init__()
        self.conv_One = DimReduceLayer(in_planes, reduce_planes, nonlinear='relu')
        self.conv_Rest = DimReduceLayer(in_planes, reduce_planes, nonlinear='relu')

        self.filter = selfAttention(reduce_planes)

    def forward(self, one, rest1, rest2):
        combined = self.conv_Rest(rest1 + rest2)
        channel_weight = self.filter(combined)
        out = self.conv_One(one) + channel_weight
        return out


class PFNET(nn.Module):
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
        super(PFNET, self).__init__()
        self.modal_number = 3
        self.loss = loss
        self.parts = 2
        self.num_classes = num_classes
        self.l2_norm = normalize()
        self.reduce_pooling_channel = 768
        self.cls_layer_channel = 256

        self.reduce_resnet_channel = 2048

        # base network
        self.backbone = nn.ModuleList(
            [
                resnet50backbone(num_classes, pretrained=True)
                for _ in range(self.modal_number)
            ]
        )
        self.reduce_layer = nn.ModuleList(
            [   
                DimReduceLayer(self.reduce_resnet_channel, self.reduce_pooling_channel, nonlinear='relu')
                for _ in range(5)
            ]
        )
        self.global_part_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))


        # fully connected layer
        self.fc_RGB = nn.ModuleList([self._fc_layer(self.reduce_pooling_channel, self.cls_layer_channel) for _ in range(self.parts)])
        self.fc_TI = nn.ModuleList([self._fc_layer(self.reduce_pooling_channel, self.cls_layer_channel) for _ in range(self.parts)])
        self.fc_NI = nn.ModuleList([self._fc_layer(self.reduce_pooling_channel, self.cls_layer_channel) for _ in range(self.parts)])
        self.fc_RT = nn.ModuleList([self._fc_layer(self.reduce_pooling_channel, self.cls_layer_channel) for _ in range(self.parts)])
        self.fc_RN = nn.ModuleList([self._fc_layer(self.reduce_pooling_channel, self.cls_layer_channel) for _ in range(self.parts)])


        # identity classification layer
        self.classifier_all = nn.Linear(2560, num_classes)
        self.classifier_RGB = nn.ModuleList([nn.Linear(self.cls_layer_channel, num_classes) for _ in range(self.parts)])
        self.classifier_TI = nn.ModuleList([nn.Linear(self.cls_layer_channel, num_classes) for _ in range(self.parts)])
        self.classifier_NI = nn.ModuleList([nn.Linear(self.cls_layer_channel, num_classes) for _ in range(self.parts)])
        self.classifier_RT = nn.ModuleList([nn.Linear(self.cls_layer_channel, num_classes) for _ in range(self.parts)])
        self.classifier_RN = nn.ModuleList([nn.Linear(self.cls_layer_channel, num_classes) for _ in range(self.parts)])


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


    def poolingLayer(self, resnetFeature_RGB, resnetFeature_NI, resnetFeature_TI, resnetFeature_RT, resnetFeature_RN):
        part_RGB = self.reduce_layer[0](self.global_part_avgpool(resnetFeature_RGB))
        part_NI = self.reduce_layer[1](self.global_part_avgpool(resnetFeature_NI))
        part_TI = self.reduce_layer[2](self.global_part_avgpool(resnetFeature_TI))
        part_RT = self.reduce_layer[3](self.global_part_avgpool(resnetFeature_RT))
        part_RN = self.reduce_layer[4](self.global_part_avgpool(resnetFeature_RN))

        return part_RGB, part_NI, part_TI, part_RT, part_RN

    def forward(self, x, return_featuremaps=False):
        # === ResNet50 layers ===
        # input shape:
        # x: [BatchSize, 3, 256, 128] * 3
        # output shape:
        # resnetFeature_RGB, resnetFeature_NI, resnetFeature_TI: [BatchSize, 2048, 16, 8]

        resnetFeature_RGB = self.backbone[0](x[0])
        resnetFeature_NI = self.backbone[1](x[1])
        resnetFeature_TI = self.backbone[2](x[2])
        # print('resnet output: ', resnetFeature_RGB.shape, resnetFeature_NI.shape, resnetFeature_TI.shape)
        if return_featuremaps:
            return resnetFeature_RGB, resnetFeature_NI, resnetFeature_TI

        resnetFeature_RT = resnetFeature_RGB + resnetFeature_TI
        resnetFeature_RN = resnetFeature_RGB + resnetFeature_NI

        # === Pooling layers ===
        # output shape:
        # afterPooling_RGB, afterPooling_NI, afterPooling_TI: [BatchSize, 2048, 1, 1]

        afterPooling_RGB, afterPooling_NI, afterPooling_TI, afterPooling_RT, afterPooling_RN = \
                self.poolingLayer(resnetFeature_RGB, resnetFeature_NI, resnetFeature_TI, resnetFeature_RT, resnetFeature_RN)
        # print('pooling output: ', afterPooling_RGB.shape, afterPooling_NI.shape, afterPooling_TI.shape)

        # === FC layers ===
        # output shape:
        # fc_RGB, fc_NI, fc_TI: [BatchSize, 512]
        # fc_all: [BatchSize, 2048]

        part_RGB = []; part_NI = []; part_TI = []; part_RT = []; part_RN = []
        for i in range(self.parts):
            part_RGB.append((afterPooling_RGB[:, :, i, :]).view((afterPooling_RGB[:, :, i, :]).size(0), -1))
            part_NI.append((afterPooling_NI[:, :, i, :]).view((afterPooling_NI[:, :, i, :]).size(0), -1))
            part_TI.append((afterPooling_TI[:, :, i, :]).view((afterPooling_TI[:, :, i, :]).size(0), -1))   
            part_RT.append((afterPooling_RT[:, :, i, :]).view((afterPooling_RT[:, :, i, :]).size(0), -1))  
            part_RN.append((afterPooling_RN[:, :, i, :]).view((afterPooling_RN[:, :, i, :]).size(0), -1))  

        # print('part output: ', len(part_RGB), part_RGB[0].shape)

        fc_RGB = []; fc_NI = []; fc_TI = []; fc_RT = []; fc_RN = []
        for i in range(self.parts):
            fc_RGB.append(self.fc_RGB[i](part_RGB[i]))
            fc_NI.append(self.fc_NI[i](part_NI[i]))
            fc_TI.append(self.fc_TI[i](part_TI[i]))
            fc_RT.append(self.fc_RT[i](part_RT[i]))
            fc_RN.append(self.fc_RN[i](part_RN[i]))
        # print('fc output: ', len(fc_RGB), fc_RGB[0].shape)

        fc_RGB_all = torch.cat(fc_RGB, dim=1)
        fc_NI_all = torch.cat(fc_NI, dim=1)
        fc_TI_all = torch.cat(fc_TI, dim=1)
        fc_RT_all = torch.cat(fc_RT, dim=1)
        fc_RN_all = torch.cat(fc_RN, dim=1)
        # print('fc_all output: ', fc_RGB_all.shape)

        fc_all = torch.cat([fc_TI_all, fc_RT_all, fc_RGB_all, fc_RN_all, fc_NI_all], dim=-1)
        # print('all output: ', fc_all.shape)


        # classifier layers
        result = []
        for i in range(self.parts):
            result.append(self.classifier_RGB[i](fc_RGB[i]))
            result.append(self.classifier_NI[i](fc_NI[i]))
            result.append(self.classifier_TI[i](fc_TI[i]))
            result.append(self.classifier_RT[i](fc_RT[i]))
            result.append(self.classifier_RN[i](fc_RN[i]))
        result.append(self.classifier_all(fc_all))

        # print('result: ', len(result))

        if not self.training:
            return fc_all
        elif self.loss == 'softmax':
            return result
        elif self.loss == 'triplet':
            return result, self.l2_norm(fc_all)
        elif self.loss == 'margin':
            return result, self.l2_norm(fc_RGB_all), self.l2_norm(fc_NI_all), self.l2_norm(fc_TI_all)
        elif self.loss == 'CMT':
            return result, self.l2_norm(fc_RGB_all), self.l2_norm(fc_NI_all), self.l2_norm(fc_TI_all), self.l2_norm(fc_all)


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


def pfnet(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PFNET(
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
