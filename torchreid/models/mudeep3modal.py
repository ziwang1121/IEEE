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
from .mudeep import MuDeep

__all__ = ['mudeep3modal']


class Mudeep3modal(nn.Module):
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
        super(Mudeep3modal, self).__init__()
        self.modal_number = 3
        self.loss = loss
        self.num_classes = num_classes

        # base network
        self.backbone = nn.ModuleList(
            [
                MuDeep(num_classes, pretrained=True)
                for _ in range(self.modal_number)
            ]
        )
        # identity classification layer

        self.classifier_RGB = nn.Linear(4096, num_classes)
        self.classifier_TI = nn.Linear(4096, num_classes)
        self.classifier_NI = nn.Linear(4096, num_classes)

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

        if not self.training:
            return fc_all
        
        # classifier layers
        result = []
        result.append(self.classifier_RGB(fc_RGB))
        result.append(self.classifier_NI(fc_NI))
        result.append(self.classifier_TI(fc_TI))
        # print('result: ', len(result))


        if self.loss == 'softmax':
            return result



def mudeep3modal(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = Mudeep3modal(
        num_classes=num_classes,
        loss=loss,
        block=None,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=6,
        reduced_dim=768,
        nonlinear='relu',
        **kwargs
    )
    return model
