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
from .resnet import resnet50_ieee, resnet50backbone
import torch
from .layers import GraphAttentionLayer, SpGraphAttentionLayer
import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
from .util import *


__all__ = ['ieee3modalPart']


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

        f_value = torch.matmul(f_part, similarity) + f_part
        # print(f_value.shape)

        # print(self.param)
        final_feat = query.unsqueeze(2) + torch.matmul(f_value, self.param.unsqueeze(0))
        # final_feat = query.unsqueeze(2) + f_value
        # print(final_feat.shape)

        return final_feat.squeeze(2)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



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
        # self.param = nn.Parameter(torch.zeros(1))
    
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
        # W_y = torch.matmul(y, self.param.unsqueeze(0))
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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        # print("1")
        return self.sigmoid(out)


class IEEE3modalPart(nn.Module):
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
        modal_number = 3
        fc_dims = [128]
        pooling_dims = 768
        super(IEEE3modalPart, self).__init__()
        self.loss = loss
        self.parts = 6
        
        self.backbone = nn.ModuleList(
            [
                resnet50_ieee(num_classes, pretrained=pretrained)
                for _ in range(3)
            ]
        )

        self.interaction = True
        self.attention = True
        self.using_REM = True
        if self.interaction:
            self.convOne = nn.ModuleList(
                [
                    DimReduceLayer(2048, 2048, nonlinear='relu')
                    for _ in range(modal_number)
                ]
            )
            self.convAvgRest = nn.ModuleList(
                [
                    DimReduceLayer(2048, 2048, nonlinear='relu')
                    for _ in range(modal_number)
                ]
            )
            if self.attention:
                self.CA = nn.ModuleList(
                    [
                        ChannelAttention(2048)
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

        
        if self.using_REM:
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
        self.fc_T = nn.ModuleList(
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
        self.classifier_T = nn.ModuleList(
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


    def crossModalInteractionModule(self, One, Rest1, Rest2, index):
        conv_feature_one = self.convOne[index](One)
        conv_feature_avgRest = self.convAvgRest[index](Rest1 + Rest2)
        if self.interaction and self.attention:
            conv_feature_avgRest1 = self.CA[index](conv_feature_avgRest) * conv_feature_avgRest + conv_feature_avgRest
            interaction_feature = conv_feature_one + conv_feature_avgRest1
        elif self.interaction and not self.attention:
            interaction_feature = conv_feature_one + conv_feature_avgRest
        return interaction_feature



    def forward(self, x, return_featuremaps=False):
        resnet_R = self.backbone[0](x[0])
        resnet_N = self.backbone[1](x[1])
        resnet_T = self.backbone[2](x[2])

        if self.interaction:
            pooling_R = self.crossModalInteractionModule(resnet_R, resnet_N, resnet_T, index=0)
            pooling_N = self.crossModalInteractionModule(resnet_N, resnet_R, resnet_T, index=1)
            pooling_T = self.crossModalInteractionModule(resnet_T, resnet_R, resnet_N, index=2)
            
            global_R = self.reduce_layer[0](self.avgpool(resnet_R))
            global_N = self.reduce_layer[1](self.avgpool(resnet_N))
            global_T = self.reduce_layer[2](self.avgpool(resnet_T))
            # print(global_T.shape)
            pooling_R = self.reduce_layer[0](self.global_avgpool(pooling_R))
            pooling_N = self.reduce_layer[1](self.global_avgpool(pooling_N))
            pooling_T = self.reduce_layer[2](self.global_avgpool(pooling_T))

            global_R = global_R.view((global_R[:, :, 0, :]).size(0), -1)
            global_N = global_N.view((global_N[:, :, 0, :]).size(0), -1)
            global_T = global_T.view((global_T[:, :, 0, :]).size(0), -1)
            # print(global_T.shape)

        else:
            global_R = self.reduce_layer[0](self.avgpool(resnet_R))
            global_N = self.reduce_layer[1](self.avgpool(resnet_N))
            global_T = self.reduce_layer[2](self.avgpool(resnet_T))


            pooling_R = self.reduce_layer[0](self.global_avgpool(resnet_R))
            pooling_N = self.reduce_layer[1](self.global_avgpool(resnet_N))
            pooling_T = self.reduce_layer[2](self.global_avgpool(resnet_T))

            global_R = global_R.view((global_R[:, :, 0, :]).size(0), -1)
            global_N = global_N.view((global_N[:, :, 0, :]).size(0), -1)
            global_T = global_T.view((global_T[:, :, 0, :]).size(0), -1)


        part_R = []; part_N = []; part_T = []
        for i in range(self.parts):
            part_R.append((pooling_R[:, :, i, :]).view((pooling_R[:, :, i, :]).size(0), -1))
            part_N.append((pooling_N[:, :, i, :]).view((pooling_N[:, :, i, :]).size(0), -1))
            part_T.append((pooling_T[:, :, i, :]).view((pooling_T[:, :, i, :]).size(0), -1))   


        if self.using_REM:
            for i in range(self.parts):
                part_R[i] = self.REM[0](part_R[i], global_R)
                part_N[i] = self.REM[1](part_N[i], global_N)
                part_T[i] = self.REM[2](part_T[i], global_T)


        fc_R = []; fc_N = []; fc_T = []
        for i in range(self.parts):
            fc_R.append(self.fc_R[i](part_R[i]))
            fc_N.append(self.fc_N[i](part_N[i]))
            fc_T.append(self.fc_T[i](part_T[i]))

        fc_R_all = torch.cat(fc_R, dim=1)
        fc_N_all = torch.cat(fc_N, dim=1)
        fc_T_all = torch.cat(fc_T, dim=1)


        fc_all = torch.cat([fc_T_all, fc_R_all, fc_N_all], dim=1)

        if not self.training:
            return fc_all

        result_R = []; result_N = []; result_T = []
        for i in range(self.parts):
            result_R.append(self.classifier_R[i](fc_R[i]))
            result_N.append(self.classifier_N[i](fc_N[i]))
            result_T.append(self.classifier_T[i](fc_T[i]))


        if self.loss == 'softmax':
            return result_R, result_N, result_T
        elif self.loss == 'triplet':
            return result_R, result_N, result_T, self.l2_norm(fc_all)
        elif self.loss == 'margin':
            return result_R, result_N, result_T, F.normalize(fc_R_all, p=2, dim=1), F.normalize(fc_N_all, p=2, dim=1), F.normalize(fc_T_all, p=2, dim=1)
        elif self.loss == 'hcloss':
            return result_R, result_N, result_T, self.l2_norm(fc_R_all), self.l2_norm(fc_N_all), self.l2_norm(fc_T_all)
        elif self.loss == 'CMT':
            return result_R, result_N, result_T, self.l2_norm(fc_R_all), self.l2_norm(fc_N_all), self.l2_norm(fc_T_all)


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


def ieee3modalPart(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = IEEE3modalPart(
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
