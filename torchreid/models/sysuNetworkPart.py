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
import torch
from .layers import GraphAttentionLayer, SpGraphAttentionLayer
import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
from .util import *


__all__ = ['sysuNetworkPart']


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


class nonLocal(nn.Module):
    def __init__(self, in_dim):
        super(nonLocal, self).__init__()
        self.conv_query = nn.Linear(in_dim, in_dim)
        self.conv_part = nn.Linear(in_dim, in_dim)
        self.conv_value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, query, part):
        f_query = self.conv_query(query).unsqueeze(1)
        # print(f_query.shape)

        f_part = self.conv_part(part).unsqueeze(2)
        # print(f_part.shape)
        f_value = self.conv_value(part).unsqueeze(2)
        energy = torch.matmul(f_query, f_part)

        similarity = self.softmax(energy)
        # print(similarity.shape)

        f_value = torch.matmul(f_part, similarity)
        # print(f_value.shape)

        # print(self.param)
        # final_feat = query.unsqueeze(2) + torch.matmul(f_value, self.param.unsqueeze(0))
        final_feat = query.unsqueeze(2) + f_value
        # print(final_feat.shape)

        return final_feat.squeeze(2)


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
        self.g = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        self.W_z = nn.Sequential(
                nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                nn.BatchNorm1d(self.in_channels)
            )
        # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

        # define theta and phi for all operations except gaussian
        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.param = nn.Parameter(torch.zeros(1))
    
    def forward(self, f_part, f_global):
        batch_size = f_global.size(0)
        
        # (N, C, THW)
        # vs
        # g_x = self.g(f_global).view(batch_size, self.inter_channels, -1)
        g_x = self.g(f_global)
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
        W_y = torch.matmul(y, self.param.unsqueeze(0))
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


class SYSUNetworkPart(nn.Module):
    def __init__(
        self,
        num_classes,
        loss,
        block,
        parts=1,
        reduced_dim=512,
        cls_dim=128,
        nonlinear='relu',
        pretrained=True,
        **kwargs
    ):
        modal_number = 2
        fc_dims = [128]
        pooling_dims = 768
        super(SYSUNetworkPart, self).__init__()
        self.loss = loss
        self.parts = 6
        
        self.backbone = nn.ModuleList(
            [
                resnet50_ieee(num_classes, pretrained=pretrained)
                for _ in range(modal_number)
            ]
        )

        self.interaction = False
        self.attention = False
        if self.interaction:
            self.convOne = nn.ModuleList(
                [
                    DimReduceLayer(2048, 2048, nonlinear='relu')
                    for _ in range(modal_number)
                ]
            )
            self.convAvgRest = nn.ModuleList(
                [
                    DimReduceLayer(4096, 2048, nonlinear='relu')
                    for _ in range(modal_number)
                ]
            )
            if self.attention:
                self.coarseGrainedAtten = nn.ModuleList(
                    [
                        selfAttention(2048)
                        for _ in range(modal_number)
                    ]
                )
        self.reduce_layer = nn.ModuleList(
                [
                    DimReduceLayer(2048, 768, nonlinear='relu')
                    for _ in range(modal_number)
                ]
            )

        self.global_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.enhancement_NL = False
        if self.enhancement_NL:
            self.REM = nn.ModuleList(
                [
                    nonLocal(768)
                    for _ in range(modal_number)
                ]
            )
        
        self.fc_R = nn.ModuleList(
            [
                self._construct_fc_layer(fc_dims, pooling_dims, None)
                for _ in range(self.parts)
            ]
        )
        self.fc_N = nn.ModuleList(
            [
                self._construct_fc_layer(fc_dims, pooling_dims, None)
                for _ in range(self.parts)
            ]
        )
        # self.fc_T = nn.ModuleList(
        #     [
        #         self._construct_fc_layer(fc_dims, pooling_dims, None)
        #         for _ in range(self.parts)
        #     ]
        # )

        # self.classifier_all = nn.Linear(fc_dims[0]*modal_number*self.parts, num_classes)
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
        # self.classifier_T = nn.ModuleList(
        #     [   
        #         nn.Linear(fc_dims[0], num_classes)
        #         for _ in range(self.parts)
        #     ]
        # )

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


    def crossModalInteractionModule(self, One, Rest1, Rest2, index):
        conv_feature_one = self.convOne[index](One)
        conv_feature_avgRest = self.convAvgRest[index](torch.cat([Rest1, Rest2], dim=1))
        if self.interaction and self.attention:
            coarse_grained_feature = self.coarseGrainedAtten[index](conv_feature_avgRest)
            interaction_feature = conv_feature_one + coarse_grained_feature
        elif self.interaction and not self.attention:
            interaction_feature = conv_feature_one + conv_feature_avgRest
        return interaction_feature


    def forward(self, x, camid):
        if camid[0] in [2, 5]:
            resnet = self.backbone[0](x)
        else:
            resnet = self.backbone[1](x)

        # if self.interaction:
        #     pooling_R = self.crossModalInteractionModule(resnet_R, resnet_N, resnet_T, index=0)
        #     pooling_N = self.crossModalInteractionModule(resnet_N, resnet_R, resnet_T, index=1)
        #     pooling_T = self.crossModalInteractionModule(resnet_T, resnet_R, resnet_N, index=2)
            
        #     global_R = self.reduce_layer[0](self.avgpool(resnet_R))
        #     global_N = self.reduce_layer[1](self.avgpool(resnet_N))
        #     global_T = self.reduce_layer[2](self.avgpool(resnet_T))
        #     # print(global_T.shape)
        #     pooling_R = self.reduce_layer[0](self.global_avgpool(pooling_R))
        #     pooling_N = self.reduce_layer[1](self.global_avgpool(pooling_N))
        #     pooling_T = self.reduce_layer[2](self.global_avgpool(pooling_T))

        #     global_R = global_R.view((global_R[:, :, 0, :]).size(0), -1)
        #     global_N = global_N.view((global_N[:, :, 0, :]).size(0), -1)
        #     global_T = global_T.view((global_T[:, :, 0, :]).size(0), -1)
        #     # print(global_T.shape)

        # else:
        if camid[0] in [2, 5]:
            pooling = self.reduce_layer[0](self.global_avgpool(resnet))
        else:
            pooling = self.reduce_layer[1](self.global_avgpool(resnet))


        part = []
        for i in range(self.parts):
            part.append((pooling[:, :, i, :]).view((pooling[:, :, i, :]).size(0), -1))
            # part_N.append((pooling_N[:, :, i, :]).view((pooling_N[:, :, i, :]).size(0), -1))
            # part_T.append((pooling_T[:, :, i, :]).view((pooling_T[:, :, i, :]).size(0), -1))   


        # if self.enhancement_NL:
        #     for i in range(self.parts):
        #         # print("111111")
        #         part_R[i] = self.REM[0](part_R[i], global_R)
        #         part_N[i] = self.REM[1](part_N[i], global_N)
        #         part_T[i] = self.REM[2](part_T[i], global_T)


        fc = []
        if camid[0] in [2, 5]:
            for i in range(self.parts):
                fc.append(self.fc_N[i](part[i]))
                # fc_N.append(self.fc_N[i](part_N[i]))
                # fc_T.append(self.fc_T[i](part_T[i]))
        else:
            for i in range(self.parts):
                fc.append(self.fc_R[i](part[i]))

        fc_all = torch.cat(fc, dim=1)
        # fc_N_all = torch.cat(fc_N, dim=1)
        # fc_T_all = torch.cat(fc_T, dim=1)

        # if self.enhancement_NL:
        #     # print("using Transformer")
        #     for i in range(self.parts):
        #         # print(fc_R_all.shape, part_R[i].shape)
        #         # print(fc_R_all[0, 100: 150], fc_N_all[0, 100: 150], fc_T_all[0, 100: 150])
        #         fc_R_all = self.REM[0](fc_R_all, part_R[i])
        #         fc_N_all = self.REM[1](fc_N_all, part_N[i])
        #         fc_T_all = self.REM[2](fc_T_all, part_T[i])
        #         # print(fc_R_all[0, 100: 150], fc_N_all[0, 100: 150], fc_T_all[0, 100: 150])
        #         # raise RuntimeError
        # # print("end", fc_R_all.shape, fc_N_all.shape, fc_T_all.shape)
        
        # fc_all = torch.cat([fc_T_all, fc_R_all, fc_N_all], dim=1)

        if not self.training:
            return fc_all

        result = []
        if camid[0] in [2, 5]:
            for i in range(self.parts):
                result.append(self.classifier_N[i](fc[i]))
        else:
            for i in range(self.parts):
                result.append(self.classifier_R[i](fc[i]))

        if self.loss == 'softmax':
            return result



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


def sysuNetworkPart(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SYSUNetworkPart(
        num_classes=num_classes,
        loss=loss,
        block=None,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=6,
        reduced_dim=768,
        nonlinear='relu',
        pretrained=pretrained,
        **kwargs
    )
    return model
