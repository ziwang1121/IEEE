from __future__ import division, absolute_import
from functools import partial
from pickle import NONE, TRUE
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
from .pcb import pcb_p4

__all__ = ['pcb3modal']


class PCB3modal(nn.Module):
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
        super(PCB3modal, self).__init__()
        self.modal_number = 3
        self.loss = loss
        self.num_classes = num_classes

        # base network
        self.backbone = nn.ModuleList(
            [
                pcb_p4(num_classes, pretrained=True)
                for _ in range(self.modal_number)
            ]
        )
        # identity classification layer

        self.classifier_RGB = nn.ModuleList(
            [
                nn.Linear(256, num_classes)
                for _ in range(4)
            ]
        )
        self.classifier_TI = nn.ModuleList(
            [
                nn.Linear(256, num_classes)
                for _ in range(4)
            ]
        )
        self.classifier_NI = nn.ModuleList(
            [
                nn.Linear(256, num_classes)
                for _ in range(4)
            ]
        )

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
        # print('resnet output: ', fc_RGB[0].shape, fc_NI[0].shape, fc_TI[0].shape)
        # print('all output: ', fc_all.shape)

        if not self.training:
            return torch.cat([fc_TI, fc_RGB, fc_NI], dim=1)

        # classifier layers
        result = []
        for i in range(4):
            result.append(self.classifier_RGB[i](fc_RGB[i]))
            result.append(self.classifier_NI[i](fc_NI[i]))
            result.append(self.classifier_TI[i](fc_TI[i]))

        # print('result: ', len(result))

        if self.loss == 'softmax':
            return result
        elif self.loss == 'triplet':
            return result



def pcb3modal(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB3modal(
        num_classes=num_classes, 
        pretrained=pretrained, 
        block=None,
        loss=loss,
        **kwargs
    )
    return model
