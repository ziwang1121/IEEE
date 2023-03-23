# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import pdb
import torch
from torch import nn
from torch.nn.modules.module import register_module_backward_hook

__all__ = [
    'MainNet'
]

from .resnet_zxp import ResNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride, 
                            #    block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet50':
            self.base  = ResNet(last_stride=last_stride,
                            #    block=Bottleneck,
                               layers=[3, 4, 6, 3])
        else:
            raise Exception('unsupported model')

        if pretrain_choice == 'imagenet':
            self.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.conv1 = nn.Conv2d(self.in_planes, num_classes, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        f = self.base.layer3(x)
        f = self.base.layer4(f)
        global_feat= self.gap(f) # B, Channel, 1, 1
        f = self.conv1(f) # B, Classnum, H, W
        # f = self.conv1(self.base(x))
        # global_feat1 = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'bnneck':
            feat = self.bottleneck(global_feat) # normalize for angular softmax
        else:
            feat = global_feat

        return feat,f

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        net_params_keys = self.state_dict().keys()
        for key in net_params_keys:
            if 'num_batches_tracked' in key:
                continue
            if key[5:] not in param_dict:
                continue
            self.state_dict()[key].copy_(param_dict[key[5:]])

class MainNet(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, loss='triplet', pretrained=True, branches=3, **kwargs):
        super(MainNet, self).__init__()
        
        self.branches = branches
        for i in range(branches):
            self.__setattr__('branch_'+str(i), Baseline(num_classes, last_stride=1, model_path=r"C:\Users\littleprince\.cache\torch\hub\checkpoints\resnet50-19c8e357.pth", neck="bnneck", neck_feat="after", model_name="resnet50", pretrain_choice="imagenet"))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes)
 
    def forward(self, inputs):
        # method in AAAI2020---------------------------------------------------------------------------------
        gf_list = [] # saving global features for final representation and loss computing
        bf_list = [] # saving basic features act as CAM feature
        for i in range(self.branches):
            global_feat, base_feat = self.__getattr__('branch_'+str(i))(inputs[i])
            base_feat = torch.sigmoid(base_feat)
            gf_list.append(global_feat)
            bf_list.append(base_feat)
        
        max_f = torch.max(torch.stack(bf_list), dim=0)[0]
        sum_f_list = []
        for i in range(self.branches):
            sum_f_list.append(bf_list[i].sum(dim=[2, 3])) # B, C
        
        max_f_rate = max_f.sum(dim=[2, 3]) / torch.unsqueeze(max_f.sum(dim=[1, 2, 3]), 1)
        sum_f = sum(sum_f_list)
        final_feat = 0
        fs = 0
        for i in range(self.branches):
            final_feat += torch.unsqueeze((sum_f_list[i]/sum_f * max_f_rate).sum(dim=1), 1) * gf_list[i]
            fs += sum_f_list[i]/sum_f * max_f_rate * sum_f_list[i]
        # method in AAAI2020---------------------------------------------------------------------------------

        if not self.training:
            return final_feat
        else:
            branch_cls_scores = []
            for i in range(self.branches):
                branch_cls_scores.append(self.classifier(gf_list[i]))
            # pdb.set_trace()
            return  branch_cls_scores, gf_list # global feature for triplet loss
            
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
