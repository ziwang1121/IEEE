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
from .hacnn import HACNN

__all__ = ['hacnn3modal']


class HACNN3modal(nn.Module):
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
        super(HACNN3modal, self).__init__()
        self.modal_number = 3
        self.loss = loss
        self.num_classes = num_classes

        # base network
        self.backbone = nn.ModuleList(
            [
                HACNN(num_classes, pretrained=True)
                for _ in range(self.modal_number)
            ]
        )
        # identity classification layer

        self.classifier_RGB_g = nn.Linear(512, num_classes)
        self.classifier_TI_g = nn.Linear(512, num_classes)
        self.classifier_NI_g = nn.Linear(512, num_classes)
        self.classifier_RGB_l = nn.Linear(512, num_classes)
        self.classifier_TI_l = nn.Linear(512, num_classes)
        self.classifier_NI_l = nn.Linear(512, num_classes)

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
        fc_RGB_all = torch.cat(fc_RGB, dim=1)
        fc_NI_all = torch.cat(fc_NI, dim=1)
        fc_TI_all = torch.cat(fc_TI, dim=1)
        fc_all = torch.cat([fc_RGB_all, fc_NI_all, fc_TI_all], dim=1)
        # print('all output: ', fc_all.shape)
        if not self.training:
            return fc_all
        # classifier layers
        result = []
        result.append(self.classifier_RGB_g(fc_RGB[0]))
        result.append(self.classifier_NI_g(fc_NI[0]))
        result.append(self.classifier_TI_g(fc_TI[0]))
        result.append(self.classifier_RGB_l(fc_RGB[1]))
        result.append(self.classifier_NI_l(fc_NI[1]))
        result.append(self.classifier_TI_l(fc_TI[1]))


        if self.loss == 'softmax':
            return result
        elif self.loss == 'triplet':
            return result, fc_all



def hacnn3modal(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = HACNN3modal(
        num_classes,
        loss,
        block=None,
        parts=6,
        reduced_dim=512,
        cls_dim=128,
        nonlinear='relu',
        **kwargs
    )
    return model
