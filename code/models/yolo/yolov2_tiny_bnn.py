# import torch
# import torch.nn as nn
# import numpy as np

# from utils.modules import Conv, reorg_layer, SpikeRelu, BinaryNeuron
# from backbone import *

import time
from builtins import isinstance
from re import X
import numpy as np
import torch
import torch.nn as nn

# from utils import box_ops

from ..basic.conv import Conv 
from ..basic.spikinglayer import SpikeRelu, BinaryNeuron, SpikeB, MemB, LTC
from ..snndarts_retrain.new_model_2d import newFeature
from ..snndarts_search.operations_2d import *
# from ..snndarts_search.decoding_formulas import network_layer_to_space

class Conv_Bn_Binary(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, groups=1, bn=True, b=3):
        super(Conv_Bn_Binary, self).__init__()
        self.bn = bn
        if padding == None:
            padding = kernel_size // 2
        self.layer = nn.Sequential()
        self.layer.add_module('cov', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
        if bn:
            self.layer.add_module('bn', nn.BatchNorm2d(out_channels))
        self.layer.add_module('spike', BinaryNeuron(b=b))

    def forward(self, x):
        output = self.layer(x)
        return output

class Conv_Bn_Spike(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, groups=1, bn=True, b=3):
        super(Conv_Bn_Spike, self).__init__()
        self.bn = bn
        if padding == None:
            padding = kernel_size // 2
        self.layer = nn.Sequential()
        self.layer.add_module('cov', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
        if bn:
            self.layer.add_module('bn', nn.BatchNorm2d(out_channels))
        self.layer.add_module('spike', SpikeB(b=b))

    def forward(self, x):
        output = self.layer(x)
        return output

class Conv_Bn_Mem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, groups=1, bn=True, b=3):
        super(Conv_Bn_Mem, self).__init__()
        self.bn = bn
        if padding == None:
            padding = kernel_size // 2
        self.layer = nn.Sequential()
        self.layer.add_module('cov', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
        if bn:
            self.layer.add_module('bn', nn.BatchNorm2d(out_channels))
        self.layer.add_module('spike', MemB(b=b))

    def forward(self, x):
        output = self.layer(x)
        return output

class Conv_Bn_LTC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, groups=1, bn=True, b=3):
        super(Conv_Bn_LTC, self).__init__()
        self.bn = bn
        if padding == None:
            padding = kernel_size // 2
        self.layer = nn.Sequential()
        self.layer.add_module('cov', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
        if bn:
            self.layer.add_module('bn', nn.BatchNorm2d(out_channels))
        self.layer.add_module('spike', LTC(b=b, in_channels=out_channels))

    def forward(self, x):
        output = self.layer(x)
        return output

class Conv_Bn_LeakyReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, groups=1, bn=True, b=3):
        super(Conv_Bn_LeakyReLu, self).__init__()
        self.bn = bn
        if padding == None:
            padding = kernel_size // 2
        self.layer = nn.Sequential()
        self.layer.add_module('cov', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bn))
        if bn:
            self.layer.add_module('bn', nn.BatchNorm2d(out_channels, affine=False))
            # self.layer.add_module('bn', nn.BatchNorm2d(out_channels))
        self.layer.add_module('spike', nn.LeakyReLU())

    def forward(self, x):
        output = self.layer(x)
        return output

class ConvLTC(nn.Module):

    '''more general discrete form of LTC'''
    def __init__(self, in_channels, out_channels, tau_input=True, taum_ini=[0.5, 0.8], usetaum=True, stream_opt=True, kernel_size=3, stride=1, padding=1, ode_unfolds=1):
        super().__init__()
       # torch.manual_seed(0)
       # torch.cuda.manual_seed(0)
        # hparams = {'use_erevin':False,'taum_ini':[.5,.8], 'nltc': 32, 'usetaum':True, 'ltcv1':True}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = self._make_layer(in_channels, out_channels, kernel_size, padding, stride)
        self.usetaum = usetaum
        self.stream_opt = stream_opt
        #初始化参数
        self.cm = nn.Parameter(0.1*torch.randn(out_channels,1,1)+1.0)
        self.vleak = nn.Parameter(0.1*torch.randn(out_channels,1,1)+1.0)
        if self.usetaum:
            self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(out_channels,1,1)+taum_ini[1])
        else:
            self.gleak = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(out_channels,1,1)+taum_ini[1])

        #self.tau_m = nn.Parameter((1.-5.)*torch.rand(out_channels,1,1)+5.)
        self.E_revin = nn.Parameter(0.1*torch.randn(out_channels,1,1)+1.0)# mean=1.0,std=0.1

        #self._ode_unfolds = torch.Tensor(ode_unfolds).cuda()
        self._epsilon = 1e-8

        self.sigmoid = nn.Sigmoid()
        self.tau_input = tau_input
        self.tanh = nn.Tanh()
        self.debug = None

        #这种初始化方法基于 Xavier (或 Glorot) 初始化策略，特别适用于激活函数是线性或者是像 tanh 这样的S型函数的网络。这种初始化方式的目的是保持输入和输出的方差接近一致。
        nn.init.xavier_normal_(self.conv[0].weight.data)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    #def init_v_pre(self, B, out_channels, H, W):
    #    self.v_pre = nn.Parameter(torch.zeros(B, self.out_channels, H, W)).cuda()

    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
    #        self.cm.data = self._clip(self.cm.data)
    #        self.gleak.data = self._clip(self.gleak.data)
        self.cm.data.clamp_(0,1000)
        if self.usetaum:
            self.tau_m.data.clamp_(0,2000)
            # self.tau_m.data.clamp_(0,1)
        else:
            self.gleak.data.clamp_(0,1000)
        # self.tau_m.data = self._clip(self.tau_m.data)

    def forward(self, inputs):
        '''
        :param inputs: (B, C_in, S, H, W)
        :param hidden_state: (hx: (B, C, S, H, W), cx: (B, C, S, H, W))
        :return: (B, C_out, H, W)
        '''
        # gleak = self.cm/self.tau_m
        # gleak.retain_grad()
        # print("cm vleak gleak erevin is_leaf",self.cm.is_leaf, self.vleak.is_leaf,self.gleak.is_leaf, self.E_revin.is_leaf)

        # print("gleak,vleak,cm",self.gleak.data,self.vleak.data,self.cm.data)
        #self.apply_weight_constraints()
        B, S, C, H, W = inputs.size()

        # v_pre = nn.Parameter(torch.zeros(B, self.out_channels, H, W)).cuda()
        outputs = []
        # print("input.size()",inputs.size()) # 1 2 10 h w
       # cm_t = self.cm / (1. / self._ode_unfolds)
        cm_t = self.cm
       # if is_train:
        #    cm_t.retain_grad()
        v_pre = torch.zeros(B, self.out_channels, H, W).to(inputs.device)
        for t in range(S):

            # wih = self.conv(self.sigmoid(inputs[:, :,t])) # wi*sig(x)+wh*sig(vpre)
            # wih = self.conv(inputs[:, :,t]) # wi*sig(x)+wh*sig(vpre)
            wih = self.conv(inputs[:, t, ...])

            # denominator = self.cm_t + self.gleak
            if self.tau_input:
                if self.usetaum:
                    numerator = (
                        self.tau_m * v_pre / (self.vleak + self.cm*self.sigmoid(wih)) + wih*self.E_revin
                    )
                    denominator = 1
                else:
                    numerator = (
                    cm_t * v_pre
                    + self.gleak * self.vleak
                    + wih*self.E_revin
                    )
                    denominator = cm_t + self.gleak + wih

            else:
                if self.usetaum:

                    numerator = (
                        self.tau_m * v_pre + wih# *self.E_revin
                        # self.tau_m * (v_pre + wih)# *self.E_revin
                    )
                    denominator = 1
                    # denominator = 1 + self.tau_m

                else:
                    numerator = (
                    cm_t * v_pre
                    + self.gleak * self.vleak
                    + wih
                    )
                    denominator = cm_t + self.gleak



            v_pre = numerator / (denominator + self._epsilon)

            # v_pre = self.tanh(v_pre)
            v_pre = self.sigmoid(v_pre)
            # v_pre = self.tanh(v_pre)
            # v_pre.retain_grad()

            outputs.append(v_pre)
            # outputs.append(torch.tanh(v_pre))
            # outputs.append(v_pre)
        self.debug = outputs[-1]
        if self.stream_opt:
            return torch.cat(outputs, 1).reshape(B, S, C, H, W)
        else:
            return outputs[-1]

class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)

class Res_Spike(nn.Module):
    expansion=1
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bn=True, b=3):
        super(Res_Spike, self).__init__()
        self.bn = bn
        if padding == None:
            padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ) 
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act1 = SpikeB(b)
        self.act2 = SpikeB(b)
    
    def forward(self, x):
        # pre_out = self.bn1(self.conv1(x))
        spike1 = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.identity(x)+self.bn2(self.conv2(spike1)))
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 76
        self.init_C = 76
        self.conv1 = nn.Conv2d(3, self.init_C, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.init_C)
        # self.relu = nn.ReLU(inplace=True)
        self.spike1 = SpikeB(3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.init_C, layers[0])
        self.layer2 = self._make_layer(block, self.init_C*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.init_C*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.init_C*8, layers[3], stride=2)

        self.connect_1 = SNN_2d(self.init_C*8, self.init_C*4, 1, 1, 0)
        self.connect_2 = SNN_2d(self.init_C*6, self.init_C*2, 1, 1, 0)
        self.connect_3 = SNN_2d(self.init_C*3, self.init_C, 1, 1, 0)

        self.up_conv1 = SNN_2d(self.init_C*4, self.init_C*2, 1, 1, 0)
        self.up_conv2 = SNN_2d(self.init_C*2, self.init_C, 1, 1, 0)

        self.smooth1 = SNN_2d(self.init_C*4, self.init_C*8, 3, 1, 1, dilation=1)
        self.smooth2 = SNN_2d(self.init_C*2, self.init_C*4, 3, 1, 1, dilation=1)
        self.smooth3 = SNN_2d(self.init_C, self.init_C*2, 3, 1, 1, dilation=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion),
        #     )

        layers = []
        layers.append(block(self.inplanes, planes, 3, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 3))

        return nn.Sequential(*layers)

    def forward(self, x, param):
        C_1 = self.conv1(x)
        C_1 = self.bn1(C_1)
        S_1 = self.spike1(C_1)
        S_2 = self.maxpool(S_1)

        C_2 = self.layer1(S_2)
        C_3 = self.layer2(C_2)
        C_4 = self.layer3(C_3)
        C_5 = self.layer4(C_4)

        upsample_1 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        upsample_2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        p1 = self.connect_1(C_5, param)
        p1_up = upsample_1(self.up_conv1(p1, param))

        p2 = self.connect_2(torch.cat([C_4, p1_up], dim=1), param)
        p2_up = upsample_2(self.up_conv2(p2, param))
        
        p3 = self.connect_3(torch.cat([C_3, p2_up], dim=1), param)

        return self.smooth1(p1, param), self.smooth2(p2, param), self.smooth3(p3, param)

        return C_3, C_4, C_5

class Res_SNN(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 cfg=None, center_sample=False, bn=True, init_channels=3, time_steps=5, spike_b=3, args=None):
        super(Res_SNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        # self.anchor_size = torch.tensor(anchor_size)
        # self.num_anchors = len(anchor_size)
        self.time_steps = time_steps

        self.feature = ResNet(Res_Spike, [2, 3, 3, 2])
        self.stride = [8, 16, 32]
        num_out = len(self.stride)
        anchor_size = cfg['anchor_size_gen1_{}'.format(num_out * 3)]
        self.anchor_list = anchor_size
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // len(self.stride), 2).float()
        self.num_anchors = self.anchor_size.size(1)
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

        self.head_det_1 = nn.Conv2d(self.feature.init_C*8, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        self.head_det_2 = nn.Conv2d(self.feature.init_C*4, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        self.head_det_3 = nn.Conv2d(self.feature.init_C*2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
    
    def create_grid(self, input_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = input_size, input_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [1, HW, 1, 2]   
            grid_xy = grid_xy[None, :, None, :].to(self.device)
            # [1, HW, 1, 2]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)

        return total_grid_xy, total_anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
    
    # def decode_xywh(self, txtytwth_pred):
    #     """
    #         Input:
    #             txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
    #         Output:
    #             xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
    #     """
    #     # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
    #     B, HW, ab_n, _ = txtytwth_pred.size()
    #     xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
    #     # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
    #     wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
    #     # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
    #     xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

    #     return xywh_pred

    # def decode_boxes(self, txtytwth_pred, requires_grad=False):
    #     """
    #         Input:
    #             txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
    #         Output:
    #             x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
    #     """
    #     # [H*W*anchor_n, 4]
    #     xywh_pred = self.decode_xywh(txtytwth_pred)

    #     # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
    #     x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
    #     x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
    #     x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
    #     x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
    #     x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

    #     return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred, index):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy[index]) * self.stride[index]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy[index]) * self.stride[index]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh[index]
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def clear_mem(self):
        for m in self.modules():
            if isinstance(m, SpikeB):
                m.clear_mem()

    def forward(self, x):
        # prediction = None
        self.clear_mem()
        C = self.num_classes
        final_y1, final_y2, final_y3 = None, None, None
        param = {'mixed_at_mem':True, 'left_or_right':'left','is_first':False}
        B, T, c, H, W = x.shape
        # x = x.reshape(B, -1, H, W)
        x = x[:,1:]
        x = x.reshape(B, 3, 3, H, W)
        # y = self.feature(x, param)
        for t in range(self.time_steps):
            if t == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            y = self.feature(x[:, t, ...],param)
            # if final_y1 == None:
            #     final_y1 = torch.zeros_like(y1)
            #     final_y2 = torch.zeros_like(y2)
            #     final_y3 = torch.zeros_like(y3)
            # else:
            #     final_y1 += y1
            #     final_y2 += y2
            #     final_y3 += y3
        # final_y1 = final_y1 / self.time_steps
        # final_y2 = final_y2 / self.time_steps
        # final_y3 = final_y3 / self.time_steps
        num_out = len(self.stride)
        if num_out == 1:
            y3 = y
            pred_s = self.head_det_3(y3)
            preds = [pred_s]
        elif num_out == 2:
            y2, y3 = y
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m]
        else:
            y1, y2, y3 = y
            pred_l = self.head_det_1(y1)
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m, pred_l]
        # final_y1 = y1
        # final_y2 = y2
        # final_y3 = y3

        # # det
        # pred_l = self.head_det_1(final_y1)
        # pred_m = self.head_det_2(final_y2)
        # pred_s = self.head_det_3(final_y3)

        # preds = [pred_s, pred_m, pred_l]
        obj_pred_list = []
        cls_pred_list = []
        reg_pred_list = []
        box_pred_list = []

        for i, pred in enumerate(preds):
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors, H, W] -> [B, H, W, num_anchors] ->  [B, HW*num_anchors, 1]
            obj_pred_i = pred[:, :self.num_anchors, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*C, H, W] -> [B, H, W, num_anchors*C] -> [B, H*W*num_anchors, C]
            cls_pred_i = pred[:, self.num_anchors:self.num_anchors*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*4, H, W] -> [B, H, W, num_anchors*4] -> [B, HW, num_anchors, 4]
            reg_pred_i = pred[:, self.num_anchors*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)
            box_pred_i = self.decode_bbox(reg_pred_i, i) / self.input_size

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            reg_pred_list.append(reg_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=1)
        cls_pred = torch.cat(cls_pred_list, dim=1)
        reg_pred = torch.cat(reg_pred_list, dim=1)
        box_pred = torch.cat(box_pred_list, dim=1)
        
        return obj_pred, cls_pred, reg_pred, box_pred

class YOLOv2Tiny_BNN(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 cfg=None, center_sample=False, bn=True, init_channels=5, time_steps=5, spike_b=3, args=None):
        super(YOLOv2Tiny_BNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh    #非极大值抑制
        self.center_sample = center_sample
        # self.anchor_size = torch.tensor(anchor_size)
        # self.num_anchors = len(anchor_size)
        self.time_steps = time_steps
        self.file = ''
        self.input_sparsity = []
        # network_arch_fea = [[[0., 0., 0.],
        #                     [1., 0., 0.],
        #                     [0., 0., 0.],
        #                     [0., 0., 0.]],
        #                     [[0., 0., 1.],
        #                     [0., 0., 0.],
        #                     [0., 0., 0.],
        #                     [0., 0., 0.]]]
        # network_path_fea = [0, 0, 1, 1, 2, 2, 2, 2, 2, 2]
        network_path_fea = [0,0,1,1,1,2,2,2,3,3]
        # network_path_fea = [1, 1, 1, 1, 1, 1, 1, 2, 2, 3]
        # network_path_fea = [0, 1, 2, 3, 3, 3, 3, 3, 3, 3]
        # cell_arch_fea = [[0, 2],
        #                 [1, 1],
        #                 [4, 2],
        #                 [2, 1],
        #                 [8, 1],
        #                 [7, 1]]

        # cell_arch_fea = [[0, 2],
        #                 [1, 2],
        #                 [2, 2],
        #                 [4, 2],
        #                 [5, 1],
        #                 [8, 2]]
        network_path_fea = np.array(network_path_fea)
        # network_arch_fea = network_layer_to_space(network_path_fea)

        # cell_arch_fea = [[0, 1],
        #                 [1, 1],
        #                 [4, 2],
        #                 [2, 1],
        #                 [8, 2],
        #                 [5, 1]]
        cell_arch_fea = [[1, 1],
                            [0, 1],
                            [3, 2],
                            [2, 1],
                            [7, 1],
                            [8, 1]]

        cell_arch_fea = np.array(cell_arch_fea)
        self.encoder = ConvLTC(init_channels, init_channels)
        self.feature = newFeature(init_channels, network_path_fea, cell_arch_fea, args=args)
        self.stride = self.feature.stride
        num_out = len(self.stride)
        anchor_size = cfg['anchor_size_gen1_{}'.format(num_out * 3)]#anchor_size_gen1_9
        self.anchor_list = anchor_size
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // len(self.stride), 2).float()
        self.num_anchors = self.anchor_size.size(1)
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        out_channel = 2 * args.fea_block_multiplier * args.fea_filter_multiplier
        ## backbone

        ## pred
        num_out = len(self.stride)
        if num_out == 1:
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        elif num_out == 2:
            self.head_det_2 = nn.Conv2d(out_channel * 2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        else:
            self.head_det_1 = nn.Conv2d(out_channel * 4, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_2 = nn.Conv2d(out_channel * 2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)

        # self.pred = nn.Conv2d(384*4, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)
        self.lsnn = SNN_2d_lsnn_front(1, 1, kernel_size=3, stride=1, padding=1,b=3)


    def create_grid(self, input_size):
        # w, h = input_size, input_size
        # # generate grid cells
        # ws, hs = w // self.stride[0], h // self.stride[0]
        # grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # # generate anchor_wh tensor
        # anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        # return grid_xy, anchor_wh

        #这里是给每个压帧后的时间片打框么？

        total_grid_xy = []
        total_anchor_wh = []
        w, h = input_size, input_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [1, HW, 1, 2]   
            grid_xy = grid_xy[None, :, None, :].to(self.device)
            # [1, HW, 1, 2]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)

        return total_grid_xy, total_anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred, index):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy[index]) * self.stride[index]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy[index]) * self.stride[index]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh[index]
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def forward(self, x):
        # prediction = None
        self.clear_mem()
        C = self.num_classes
        final_y1, final_y2, final_y3 = None, None, None
        param = {'mixed_at_mem':True, 'left_or_right':'left','is_first':False}
        B, T, c, H, W = x.shape
        """ if shape is not (B,3,3,H,W)
        x = x.reshape(B, -1, H, W)
        # y = self.feature(x, param)
        # x = self.encoder(x)
        # print('lsnn x in',x.shape)
        # x = self.lsnn(x)
        # x = x.reshape(B, -1, H, W)
        # x = x.reshape(B, 2, 5, H, W)
        x = x[:, 1:, ...]
        x = x.reshape(B, 3, 3, H, W)
        # # x = x[:,1:]
        # # x = x.reshape(B, 3, 3, H, W)
        B, T, c, H, W = x.shape
        # print('lsnn x out',x.shape)
        """
        

        for t in range(self.time_steps):
            self.input_sparsity.append(x[:, t, ...].detach().cpu().numpy())
            if t == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            # self.feature(x[:, t, ...],param)
            # x1 = self.down_sample1(self.conv1(x[:, t, ...]))
            # x2 = self.down_sample2(self.conv2(x1))
            # x3 = self.down_sample3(self.conv3(x2))
            # x4 = self.down_sample4(self.conv4(x3))
            # x5 = self.down_sample5(self.conv5(x4))
            # x6 = self.down_sample6(self.zero_pad(self.conv6(x5)))

            # x7 = self.conv7(x6)
            # x8 = self.conv8(x7)
            # pred = self.pred(x8)
            y = self.feature(x[:, t, ...],param)
            # if final_y1 == None:
            #     final_y1 = torch.zeros_like(y1)``
            #     final_y2 = torch.zeros_like(y2)
            #     final_y3 = torch.zeros_like(y3)
            # else:
            #     final_y1 += y1
            #     final_y2 += y2
            #     final_y3 += y3
        # final_y1 = final_y1 / self.time_steps
        # final_y2 = final_y2 / self.time_steps
        # final_y3 = final_y3 / self.time_steps
        num_out = len(self.stride)
        if num_out == 1:
            y3 = y
            pred_s = self.head_det_3(y3)
            preds = [pred_s]
        elif num_out == 2:
            y2, y3 = y
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m]
        else:
            y1, y2, y3 = y
            pred_l = self.head_det_1(y1)
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m, pred_l]
        # final_y1 = y1
        # final_y2 = y2
        # final_y3 = y3

        # # det
        # pred_l = self.head_det_1(final_y1)
        # pred_m = self.head_det_2(final_y2)
        # pred_s = self.head_det_3(final_y3)

        # preds = [pred_s, pred_m, pred_l]
        obj_pred_list = []
        cls_pred_list = []
        reg_pred_list = []
        box_pred_list = []

        for i, pred in enumerate(preds):
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors, H, W] -> [B, H, W, num_anchors] ->  [B, HW*num_anchors, 1]
            obj_pred_i = pred[:, :self.num_anchors, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*C, H, W] -> [B, H, W, num_anchors*C] -> [B, H*W*num_anchors, C]
            cls_pred_i = pred[:, self.num_anchors:self.num_anchors*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*4, H, W] -> [B, H, W, num_anchors*4] -> [B, HW, num_anchors, 4]
            reg_pred_i = pred[:, self.num_anchors*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)
            box_pred_i = self.decode_bbox(reg_pred_i, i) / self.input_size

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            reg_pred_list.append(reg_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=1)
        cls_pred = torch.cat(cls_pred_list, dim=1)
        reg_pred = torch.cat(reg_pred_list, dim=1)
        box_pred = torch.cat(box_pred_list, dim=1)
        
        return obj_pred, cls_pred, reg_pred, box_pred

    def set_mem_keys(self, mem_keys):
        self.mem_keys = mem_keys
    
    def clear_mem(self):
        for key in self.mem_keys:
            exec('self.%s.mem=None'%key)
        for m in self.modules():
            if isinstance(m, SNN_2d) or isinstance(m, SNN_2d_lsnn) or isinstance(m, SNN_2d_thresh) or isinstance(m, Mem_Relu):
                m.mem = None

    def stream_forward_new(self, x, file):
        C = self.num_classes
        param = {'mixed_at_mem':True, 'left_or_right':'left','is_first':False}
        # x = x.reshape(B, -1, H, W)
        # y = self.feature(x, param)
        # x = self.encoder(x)
        # print('lsnn x in',x.shape)
        # x = self.lsnn(x)
        B, T, c, H, W = x.shape
        x = x[:,1:]
        x = x.reshape(B, 3, 3, H, W)
        B, T, c, H, W = x.shape
        # print('lsnn x out',x.shape)
        pred_s_list = []
        pred_m_list = []
        pred_l_list = []

        for i in range(B):
            if self.file != file[i]:
                print('different file')
                self.file = file[i]
                self.clear_mem()
                param['is_first'] = True
            else:
                param['is_first'] = False
            # if t == 0:
            #     param['is_first'] = True
            # else:
            #     param['is_first'] = False
            start_time = time.time()
            for t in range(T):
                self.input_sparsity.append(torch.mean(torch.abs(x[i, t, ...])).detach().cpu().numpy())
                y = self.feature(x[i, t, ...].unsqueeze(0),param)
            # print('3 stack time:', (time.time()-start_time)/3)
            # print('!!!',x[:, t, ...])
            # # print('```',summary())
            # summary(self.feature, input_size=(1, 5, x[:, t, ...].shape[-2], x[:, t, ...].shape[-1]))
            # input()
            # flops, params = get_model_complexity_info( self.feature, (5, x[:, t, ...].shape[-2], x[:, t, ...].shape[-1]), as_strings=True, print_per_layer_stat=True)
            # input()
            # if final_y1 == None:
            #     final_y1 = torch.zeros_like(y1)``
            #     final_y2 = torch.zeros_like(y2)
            #     final_y3 = torch.zeros_like(y3)
            # else:
            #     final_y1 += y1
            #     final_y2 += y2
            #     final_y3 += y3
        # final_y1 = final_y1 / self.time_steps
        # final_y2 = final_y2 / self.time_steps
        # final_y3 = final_y3 / self.time_steps
            num_out = len(self.stride)
            if num_out == 1:
                y3 = y
                pred_s = self.head_det_3(y3)
                pred_s_list.append(pred_s)
            elif num_out == 2:
                y2, y3 = y
                pred_m = self.head_det_2(y2)
                pred_s = self.head_det_3(y3)
                pred_m_list.append(pred_m)
                pred_s_list.append(pred_s)
            else:
                y1, y2, y3 = y
                # print(y1.shape, y2.shape, y3.shape)
                pred_l = self.head_det_1(y1)
                pred_m = self.head_det_2(y2)
                pred_s = self.head_det_3(y3)
                pred_l_list.append(pred_l)
                pred_m_list.append(pred_m)
                pred_s_list.append(pred_s)
        preds = []
        obj_pred_list = []
        cls_pred_list = []
        reg_pred_list = []
        box_pred_list = []
        if len(pred_s_list) != 0:
            preds.append(torch.cat(pred_s_list, dim=0))
        if len(pred_m_list) != 0:
            preds.append(torch.cat(pred_m_list, dim=0))
        if len(pred_l_list) != 0:
            preds.append(torch.cat(pred_l_list, dim=0))

        for i, pred in enumerate(preds):
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors, H, W] -> [B, H, W, num_anchors] ->  [B, HW*num_anchors, 1]
            obj_pred_i = pred[:, :self.num_anchors, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*C, H, W] -> [B, H, W, num_anchors*C] -> [B, H*W*num_anchors, C]
            cls_pred_i = pred[:, self.num_anchors:self.num_anchors*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*4, H, W] -> [B, H, W, num_anchors*4] -> [B, HW, num_anchors, 4]
            reg_pred_i = pred[:, self.num_anchors*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)
            box_pred_i = self.decode_bbox(reg_pred_i, i) / self.input_size

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            reg_pred_list.append(reg_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=1)
        cls_pred = torch.cat(cls_pred_list, dim=1)
        reg_pred = torch.cat(reg_pred_list, dim=1)
        box_pred = torch.cat(box_pred_list, dim=1)
        
        return obj_pred, cls_pred, reg_pred, box_pred


    def stream_forward(self, x, file):
        # prediction = None
        C = self.num_classes
        param = {'mixed_at_mem':True, 'left_or_right':'left','is_first':False}
        B, T, c, H, W = x.shape
        # x = x.reshape(B, -1, H, W)
        # y = self.feature(x, param)
        # x = self.encoder(x)
        # print('lsnn x in',x.shape)
        # x = self.lsnn(x)
        x = x[:,1:]
        x = x.reshape(B, 3, 3, H, W)
        # print('lsnn x out',x.shape)
        pred_s_list = []
        pred_m_list = []
        pred_l_list = []

        for t in range(B):
            if self.file != file[t]:
                self.file = file
                self.clear_mem()
            # if t == 0:
            #     param['is_first'] = True
            # else:
            #     param['is_first'] = False
            # self.feature(x[:, t, ...],param)
            # x1 = self.down_sample1(self.conv1(x[:, t, ...]))
            # x2 = self.down_sample2(self.conv2(x1))
            # x3 = self.down_sample3(self.conv3(x2))
            # x4 = self.down_sample4(self.conv4(x3))
            # x5 = self.down_sample5(self.conv5(x4))
            # x6 = self.down_sample6(self.zero_pad(self.conv6(x5)))

            # x7 = self.conv7(x6)
            # x8 = self.conv8(x7)
            # pred = self.pred(x8)
            # print(x[t, -1, ...].unsqueeze(0).shape)
            # input()
            y = self.feature(x[t, -1, ...].unsqueeze(0),param)
            
            # print('!!!',x[:, t, ...])
            # # print('```',summary())
            # summary(self.feature, input_size=(1, 5, x[:, t, ...].shape[-2], x[:, t, ...].shape[-1]))
            # input()
            # flops, params = get_model_complexity_info( self.feature, (5, x[:, t, ...].shape[-2], x[:, t, ...].shape[-1]), as_strings=True, print_per_layer_stat=True)
            # input()
            # if final_y1 == None:
            #     final_y1 = torch.zeros_like(y1)``
            #     final_y2 = torch.zeros_like(y2)
            #     final_y3 = torch.zeros_like(y3)
            # else:
            #     final_y1 += y1
            #     final_y2 += y2
            #     final_y3 += y3
        # final_y1 = final_y1 / self.time_steps
        # final_y2 = final_y2 / self.time_steps
        # final_y3 = final_y3 / self.time_steps
            num_out = len(self.stride)
            if num_out == 1:
                y3 = y
                pred_s = self.head_det_3(y3)
                pred_s_list.append(pred_s)
            elif num_out == 2:
                y2, y3 = y
                pred_m = self.head_det_2(y2)
                pred_s = self.head_det_3(y3)
                pred_m_list.append(pred_m)
                pred_s_list.append(pred_s)
            else:
                y1, y2, y3 = y
                # print(y1.shape, y2.shape, y3.shape)
                pred_l = self.head_det_1(y1)
                pred_m = self.head_det_2(y2)
                pred_s = self.head_det_3(y3)
                pred_l_list.append(pred_l)
                pred_m_list.append(pred_m)
                pred_s_list.append(pred_s)
        # final_y1 = y1
        # final_y2 = y2
        # final_y3 = y3

        # # det
        # pred_l = self.head_det_1(final_y1)
        # pred_m = self.head_det_2(final_y2)
        # pred_s = self.head_det_3(final_y3)

        # preds = [pred_s, pred_m, pred_l]
        preds = []
        obj_pred_list = []
        cls_pred_list = []
        reg_pred_list = []
        box_pred_list = []
        if len(pred_s_list) != 0:
            preds.append(torch.cat(pred_s_list, dim=0))
        if len(pred_m_list) != 0:
            preds.append(torch.cat(pred_m_list, dim=0))
        if len(pred_l_list) != 0:
            preds.append(torch.cat(pred_l_list, dim=0))

        for i, pred in enumerate(preds):
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors, H, W] -> [B, H, W, num_anchors] ->  [B, HW*num_anchors, 1]
            obj_pred_i = pred[:, :self.num_anchors, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*C, H, W] -> [B, H, W, num_anchors*C] -> [B, H*W*num_anchors, C]
            cls_pred_i = pred[:, self.num_anchors:self.num_anchors*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*4, H, W] -> [B, H, W, num_anchors*4] -> [B, HW, num_anchors, 4]
            reg_pred_i = pred[:, self.num_anchors*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)
            box_pred_i = self.decode_bbox(reg_pred_i, i) / self.input_size

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            reg_pred_list.append(reg_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=1)
        cls_pred = torch.cat(cls_pred_list, dim=1)
        reg_pred = torch.cat(reg_pred_list, dim=1)
        box_pred = torch.cat(box_pred_list, dim=1)
        
        return obj_pred, cls_pred, reg_pred, box_pred

class YOLOv2Tiny_BNN_RGB(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, center_sample=False, bn=True, init_channels=3, time_steps=5, spike_b=3, args=None):
        super(YOLOv2Tiny_BNN_RGB, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        # self.anchor_size = torch.tensor(anchor_size)
        # self.num_anchors = len(anchor_size)
        self.time_steps = time_steps


        # network_arch_fea = [[[0., 0., 0.],
        #                     [1., 0., 0.],
        #                     [0., 0., 0.],
        #                     [0., 0., 0.]],
        #                     [[0., 0., 1.],
        #                     [0., 0., 0.],
        #                     [0., 0., 0.],
        #                     [0., 0., 0.]]]
        # network_path_fea = [0, 0, 1, 1, 2, 2, 2, 2, 2, 2]
        network_path_fea = [0,0,1,1,1,2,2,2,3,3]
        # network_path_fea = [1, 1, 1, 1, 1, 1, 1, 2, 2, 3]
        # network_path_fea = [0, 1, 2, 3, 3, 3, 3, 3, 3, 3]
        # cell_arch_fea = [[0, 2],
        #                 [1, 1],
        #                 [4, 2],
        #                 [2, 1],
        #                 [8, 1],
        #                 [7, 1]]

        # cell_arch_fea = [[0, 2],
        #                 [1, 2],
        #                 [2, 2],
        #                 [4, 2],
        #                 [5, 1],
        #                 [8, 2]]
        network_path_fea = np.array(network_path_fea)
        # network_arch_fea = network_layer_to_space(network_path_fea)

        # cell_arch_fea = [[0, 1],
        #                 [1, 1],
        #                 [4, 2],
        #                 [2, 1],
        #                 [8, 2],
        #                 [5, 1]]
        cell_arch_fea = [[1, 1],
                            [0, 1],
                            [3, 2],
                            [2, 1],
                            [7, 1],
                            [8, 1]]

        cell_arch_fea = np.array(cell_arch_fea)

        self.feature = newFeature(init_channels, network_path_fea, cell_arch_fea, args=args)
        self.stride = self.feature.stride
        num_out = len(self.stride)
        # anchor_size = cfg['anchor_size_gen1_{}'.format(num_out * 3)]
        # self.anchor_list = anchor_size
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // len(self.stride), 2).float()
        self.num_anchors = self.anchor_size.size(1)
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        out_channel = 2 * args.fea_block_multiplier * args.fea_filter_multiplier
        ## backbone

        ## pred
        num_out = len(self.stride)
        if num_out == 1:
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        elif num_out == 2:
            self.head_det_2 = nn.Conv2d(out_channel * 2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        else:
            self.head_det_1 = nn.Conv2d(out_channel * 4, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_2 = nn.Conv2d(out_channel * 2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)

        # self.pred = nn.Conv2d(384*4, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

    

    def create_grid(self, input_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = input_size, input_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [1, HW, 1, 2]   
            grid_xy = grid_xy[None, :, None, :].to(self.device)
            # [1, HW, 1, 2]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)

        return total_grid_xy, total_anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred, index):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy[index]) * self.stride[index]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy[index]) * self.stride[index]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh[index]
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def clear_mem(self):
        for m in self.modules():
            if isinstance(m, SpikeB):
                m.clear_mem()

    def forward(self, x):
        # prediction = None
        self.clear_mem()
        C = self.num_classes
        final_y1, final_y2, final_y3 = None, None, None
        param = {'mixed_at_mem':True, 'left_or_right':'left','is_first':False}
        B, c, H, W = x.shape
        # x = x.reshape(B, -1, H, W)
        # y = self.feature(x, param)
        for t in range(self.time_steps):
            if t == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            y = self.feature(x,param)
            # if final_y1 == None:
            #     final_y1 = torch.zeros_like(y1)
            #     final_y2 = torch.zeros_like(y2)
            #     final_y3 = torch.zeros_like(y3)
            # else:
            #     final_y1 += y1
            #     final_y2 += y2
            #     final_y3 += y3
        # final_y1 = final_y1 / self.time_steps
        # final_y2 = final_y2 / self.time_steps
        # final_y3 = final_y3 / self.time_steps
        num_out = len(self.stride)
        if num_out == 1:
            y3 = y
            pred_s = self.head_det_3(y3)
            preds = [pred_s]
        elif num_out == 2:
            y2, y3 = y
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m]
        else:
            y1, y2, y3 = y
            pred_l = self.head_det_1(y1)
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m, pred_l]
        # final_y1 = y1
        # final_y2 = y2
        # final_y3 = y3

        # # det
        # pred_l = self.head_det_1(final_y1)
        # pred_m = self.head_det_2(final_y2)
        # pred_s = self.head_det_3(final_y3)

        # preds = [pred_s, pred_m, pred_l]
        obj_pred_list = []
        cls_pred_list = []
        reg_pred_list = []
        box_pred_list = []

        for i, pred in enumerate(preds):
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors, H, W] -> [B, H, W, num_anchors] ->  [B, HW*num_anchors, 1]
            obj_pred_i = pred[:, :self.num_anchors, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*C, H, W] -> [B, H, W, num_anchors*C] -> [B, H*W*num_anchors, C]
            cls_pred_i = pred[:, self.num_anchors:self.num_anchors*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*4, H, W] -> [B, H, W, num_anchors*4] -> [B, HW, num_anchors, 4]
            reg_pred_i = pred[:, self.num_anchors*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)
            box_pred_i = self.decode_bbox(reg_pred_i, i) / self.input_size

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            reg_pred_list.append(reg_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=1)
        cls_pred = torch.cat(cls_pred_list, dim=1)
        reg_pred = torch.cat(reg_pred_list, dim=1)
        box_pred = torch.cat(box_pred_list, dim=1)
        
        return obj_pred, cls_pred, reg_pred, box_pred

class YOLOv2Tiny_SNN_BNN(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, center_sample=False, bn=True, init_channels=5, time_steps=5, spike_b=3):
        super(YOLOv2Tiny_SNN_BNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = [32]
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        self.time_steps = time_steps

        self.conv1 = Conv_Bn_Spike(init_channels, 16, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample1 = Conv_Bn_Binary(16, 16, 2, 2, 0, bn=False, b=spike_b)
        self.conv2 = Conv_Bn_Binary(16, 32, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample2 = Conv_Bn_Binary(32, 32, 2, 2, 0, bn=False, b=spike_b)
        self.conv3 = Conv_Bn_Binary(32, 64, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample3 = Conv_Bn_Binary(64, 64, 2, 2, 0, bn=False, b=spike_b)
        self.conv4 = Conv_Bn_Binary(64, 128, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample4 = Conv_Bn_Binary(128, 128, 2, 2, 0, bn=False, b=spike_b)
        self.conv5 = Conv_Bn_Binary(128, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample5 = Conv_Bn_Binary(256, 256, 2, 2, 0, bn=False, b=spike_b)
        self.conv6 = Conv_Bn_Binary(256, 512, 3, 1, 1, bn=bn, b=spike_b)
        self.zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.down_sample6 = Conv_Bn_Binary(512, 512, 2, 1, 0, bn=False, b=spike_b)
        self.conv7 = Conv_Bn_Binary(512, 1024, 3, 1, 1, bn=bn, b=spike_b)
        self.conv8 = Conv_Bn_Binary(1024, 512, 3, 1, 1, bn=bn, b=spike_b)

        self.pred = nn.Conv2d(512, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

    

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride[0], h // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def clear_mem(self):
        for m in self.modules():
            if isinstance(m, SpikeB):
                m.clear_mem()

    def forward(self, x):
        prediction = None
        self.clear_mem()
        for t in range(self.time_steps):
            x1 = self.down_sample1(self.conv1(x[:, t, ...]))
            x2 = self.down_sample2(self.conv2(x1))
            x3 = self.down_sample3(self.conv3(x2))
            x4 = self.down_sample4(self.conv4(x3))
            x5 = self.down_sample5(self.conv5(x4))
            x6 = self.down_sample6(self.zero_pad(self.conv6(x5)))

            x7 = self.conv7(x6)
            x8 = self.conv8(x7)
            pred = self.pred(x8)
            if prediction == None:
                prediction = torch.zeros_like(pred)
            prediction += pred
        
        # 累加时间维度
        prediction = prediction / self.time_steps

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W*num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous().view(B, H * W, self.num_anchors, 4)
        # [B, H*W*num_anchor, 4]
        x1y1x2y2_pred = self.decode_bbox(txtytwth_pred) / self.input_size

        return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred

class YOLO_SNN(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 cfg=None, center_sample=False, bn=True, init_channels=5, time_steps=5, spike_b=3, args=None):
        super(YOLO_SNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        # self.anchor_size = torch.tensor(anchor_size)
        # self.num_anchors = len(anchor_size)
        self.time_steps = time_steps

        ## backbone
        self.conv1 = Conv_Bn_Spike(init_channels, 16, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample1 = Conv_Bn_Spike(16, 16, 2, 2, 0, bn=False, b=spike_b)
        # self.down_sample1 = nn.MaxPool2d(2, 2)
        self.conv2 = Res_Spike(16, 32, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample2 = Conv_Bn_Spike(32, 32, 2, 2, 0, bn=False, b=spike_b)
        # self.down_sample2 = nn.MaxPool2d(2, 2)
        self.conv3 = Res_Spike(32, 64, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample3 = Conv_Bn_Spike(64, 64, 2, 2, 0, bn=False, b=spike_b)
        # self.down_sample3 = nn.MaxPool2d(2, 2)
        self.conv4 = Res_Spike(64, 128, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample4 = Conv_Bn_Spike(128, 128, 2, 2, 0, bn=False, b=spike_b)
        # self.down_sample4 = nn.MaxPool2d(2, 2)
        self.conv5 = Res_Spike(128, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample5 = Conv_Bn_Spike(256, 256, 2, 2, 0, bn=False, b=spike_b)
        # self.down_sample5 = nn.MaxPool2d(2, 2)
        self.conv6 = Res_Spike(256, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample6 = Conv_Bn_Spike(256, 256, 2, 2, 0, bn=False, b=spike_b)
        # self.down_sample6 = nn.MaxPool2d(2, 2)
        self.conv7 = Res_Spike(256, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample7 = Conv_Bn_Spike(256, 256, 2, 2, 0, bn=False, b=spike_b)
        # self.conv8 = Conv_Bn_Spike(1024, 2048, 3, 1, 1, bn=bn, b=spike_b)
        # self.down_sample8 = Conv_Bn_Spike(2048, 2048, 2, 2, 0, bn=False, b=spike_b)

        self.stride = [32, 64, 128]
        num_out = len(self.stride)
        anchor_size = cfg['anchor_size_gen1_{}'.format(num_out * 3)]
        self.anchor_list = anchor_size
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // len(self.stride), 2).float()
        self.num_anchors = self.anchor_size.size(1)
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        out_channel = 2 * args.fea_block_multiplier * args.fea_filter_multiplier
        ## backbone

        ## pred
        num_out = len(self.stride)
        self.head_det_1 = nn.Conv2d(256, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        self.head_det_2 = nn.Conv2d(256, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        self.head_det_3 = nn.Conv2d(256, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)

        # self.pred = nn.Conv2d(384*4, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

    

    def create_grid(self, input_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = input_size, input_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [1, HW, 1, 2]   
            grid_xy = grid_xy[None, :, None, :].to(self.device)
            # [1, HW, 1, 2]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)

        return total_grid_xy, total_anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred, index):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy[index]) * self.stride[index]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy[index]) * self.stride[index]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh[index]
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def clear_mem(self):
        for m in self.modules():
            if isinstance(m, SpikeB):
                m.clear_mem()

    def forward(self, x):
        # prediction = None
        self.clear_mem()
        C = self.num_classes
        final_y1, final_y2, final_y3 = None, None, None
        param = {'mixed_at_mem':True, 'left_or_right':'left','is_first':False}
        B, T, c, H, W = x.shape
        # x = x.reshape(B, -1, H, W)
        # y = self.feature(x, param)
        for t in range(self.time_steps):
            y = self.down_sample1(self.conv1(x[:, t, ...]))
            y = self.down_sample2(self.conv2(y))
            y = self.down_sample3(self.conv3(y))
            y = self.down_sample4(self.conv4(y))
            y1 = self.down_sample5(self.conv5(y))
            y2 = self.down_sample6(self.conv6(y1))
            y3 = self.down_sample7(self.conv7(y2))
            # y3 = self.down_sample8(self.conv8(y2))
            # y4 = self.conv7(y3)
        pred_s = self.head_det_1(y1)
        pred_m = self.head_det_2(y2)
        pred_l = self.head_det_3(y3)
        preds = [pred_s, pred_m, pred_l]
        obj_pred_list = []
        cls_pred_list = []
        reg_pred_list = []
        box_pred_list = []

        for i, pred in enumerate(preds):
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors, H, W] -> [B, H, W, num_anchors] ->  [B, HW*num_anchors, 1]
            obj_pred_i = pred[:, :self.num_anchors, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*C, H, W] -> [B, H, W, num_anchors*C] -> [B, H*W*num_anchors, C]
            cls_pred_i = pred[:, self.num_anchors:self.num_anchors*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*4, H, W] -> [B, H, W, num_anchors*4] -> [B, HW, num_anchors, 4]
            reg_pred_i = pred[:, self.num_anchors*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)
            box_pred_i = self.decode_bbox(reg_pred_i, i) / self.input_size

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            reg_pred_list.append(reg_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=1)
        cls_pred = torch.cat(cls_pred_list, dim=1)
        reg_pred = torch.cat(reg_pred_list, dim=1)
        box_pred = torch.cat(box_pred_list, dim=1)
        
        return obj_pred, cls_pred, reg_pred, box_pred

class YOLOv2Tiny_SNNM_ANN(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, center_sample=False, bn=True, init_channels=5, time_steps=5, spike_b=3):
        super(YOLOv2Tiny_SNNM_ANN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = [32]
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        self.time_steps = time_steps

        self.conv1 = Conv_Bn_Mem(init_channels, 16, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample1 = Conv_Bn_LeakyReLu(16, 16, 2, 2, 0, bn=False, b=spike_b)
        self.conv2 = Conv_Bn_LeakyReLu(16, 32, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample2 = Conv_Bn_LeakyReLu(32, 32, 2, 2, 0, bn=False, b=spike_b)
        self.conv3 = Conv_Bn_LeakyReLu(32, 64, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample3 = Conv_Bn_LeakyReLu(64, 64, 2, 2, 0, bn=False, b=spike_b)
        self.conv4 = Conv_Bn_LeakyReLu(64, 128, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample4 = Conv_Bn_LeakyReLu(128, 128, 2, 2, 0, bn=False, b=spike_b)
        self.conv5 = Conv_Bn_LeakyReLu(128, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample5 = Conv_Bn_LeakyReLu(256, 256, 2, 2, 0, bn=False, b=spike_b)
        self.conv6 = Conv_Bn_LeakyReLu(256, 512, 3, 1, 1, bn=bn, b=spike_b)
        self.zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.down_sample6 = Conv_Bn_LeakyReLu(512, 512, 2, 1, 0, bn=False, b=spike_b)
        self.conv7 = Conv_Bn_LeakyReLu(512, 1024, 3, 1, 1, bn=bn, b=spike_b)
        self.conv8 = Conv_Bn_LeakyReLu(1024, 512, 3, 1, 1, bn=bn, b=spike_b)

        self.pred = nn.Conv2d(512, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

    

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride[0], h // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def clear_mem(self):
        for m in self.modules():
            if isinstance(m, MemB):
                m.clear_mem()

    def forward(self, x):
        prediction = None
        self.clear_mem()
        for t in range(self.time_steps):
            x1 = self.down_sample1(self.conv1(x[:, t, ...]))
            x2 = self.down_sample2(self.conv2(x1))
            x3 = self.down_sample3(self.conv3(x2))
            x4 = self.down_sample4(self.conv4(x3))
            x5 = self.down_sample5(self.conv5(x4))
            x6 = self.down_sample6(self.zero_pad(self.conv6(x5)))

            x7 = self.conv7(x6)
            x8 = self.conv8(x7)
            pred = self.pred(x8)
            if prediction == None:
                prediction = torch.zeros_like(pred)
            prediction += pred
        
        # 累加时间维度
        prediction = prediction / self.time_steps

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W*num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous().view(B, H * W, self.num_anchors, 4)
        # [B, H*W*num_anchor, 4]
        # 该步将原本的坐标预测从相对于grid的(0,1)数据准换为相对于整个图片的(0,1)数据
        x1y1x2y2_pred = self.decode_bbox(txtytwth_pred) / self.input_size

        return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred


class YOLOv2Tiny_SNN_ANN(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, center_sample=False, bn=True, init_channels=5, time_steps=5, spike_b=3):
        super(YOLOv2Tiny_SNN_ANN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = [32]
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        self.time_steps = time_steps

        self.conv1 = Conv_Bn_Spike(init_channels, 16, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample1 = Conv_Bn_LeakyReLu(16, 16, 2, 2, 0, bn=False, b=spike_b)
        self.conv2 = Conv_Bn_LeakyReLu(16, 32, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample2 = Conv_Bn_LeakyReLu(32, 32, 2, 2, 0, bn=False, b=spike_b)
        self.conv3 = Conv_Bn_LeakyReLu(32, 64, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample3 = Conv_Bn_LeakyReLu(64, 64, 2, 2, 0, bn=False, b=spike_b)
        self.conv4 = Conv_Bn_LeakyReLu(64, 128, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample4 = Conv_Bn_LeakyReLu(128, 128, 2, 2, 0, bn=False, b=spike_b)
        self.conv5 = Conv_Bn_LeakyReLu(128, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample5 = Conv_Bn_LeakyReLu(256, 256, 2, 2, 0, bn=False, b=spike_b)
        self.conv6 = Conv_Bn_LeakyReLu(256, 512, 3, 1, 1, bn=bn, b=spike_b)
        self.zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.down_sample6 = Conv_Bn_LeakyReLu(512, 512, 2, 1, 0, bn=False, b=spike_b)
        # self.conv7 = Conv_Bn_LeakyReLu(512, 1024, 3, 1, 1, bn=bn, b=spike_b)
        # self.conv8 = Conv_Bn_LeakyReLu(1024, 512, 3, 1, 1, bn=bn, b=spike_b)

        self.pred = nn.Conv2d(512, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

    

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride[0], h // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def clear_mem(self):
        for m in self.modules():
            if isinstance(m, SpikeB):
                m.clear_mem()

    def forward(self, x):
        prediction = None
        self.clear_mem()
        for t in range(self.time_steps):
            x1 = self.down_sample1(self.conv1(x[:, t, ...]))
            x2 = self.down_sample2(self.conv2(x1))
            x3 = self.down_sample3(self.conv3(x2))
            x4 = self.down_sample4(self.conv4(x3))
            x5 = self.down_sample5(self.conv5(x4))
            x6 = self.down_sample6(self.zero_pad(self.conv6(x5)))

            # x7 = self.conv7(x6)
            # x8 = self.conv8(x7)
            pred = self.pred(x6)
            if prediction == None:
                prediction = torch.zeros_like(pred)
            prediction += pred
        
        # 累加时间维度
        prediction = prediction / self.time_steps

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W*num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous().view(B, H * W, self.num_anchors, 4)
        # [B, H*W*num_anchor, 4]
        # 该步将原本的坐标预测从相对于grid的(0,1)数据准换为相对于整个图片的(0,1)数据
        x1y1x2y2_pred = self.decode_bbox(txtytwth_pred) / self.input_size

        return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred

class YOLOv2Tiny_ANN(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, center_sample=False, bn=True, init_channels=5, time_steps=5, spike_b=3):
        super(YOLOv2Tiny_ANN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = [32]
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        self.time_steps = time_steps

        self.conv1 = Conv_Bn_LeakyReLu(init_channels * time_steps, 16, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample1 = Conv_Bn_LeakyReLu(16, 16, 2, 2, 0, bn=False, b=spike_b)
        self.conv2 = Conv_Bn_LeakyReLu(16, 32, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample2 = Conv_Bn_LeakyReLu(32, 32, 2, 2, 0, bn=False, b=spike_b)
        self.conv3 = Conv_Bn_LeakyReLu(32, 64, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample3 = Conv_Bn_LeakyReLu(64, 64, 2, 2, 0, bn=False, b=spike_b)
        self.conv4 = Conv_Bn_LeakyReLu(64, 128, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample4 = Conv_Bn_LeakyReLu(128, 128, 2, 2, 0, bn=False, b=spike_b)
        self.conv5 = Conv_Bn_LeakyReLu(128, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample5 = Conv_Bn_LeakyReLu(256, 256, 2, 2, 0, bn=False, b=spike_b)
        self.conv6 = Conv_Bn_LeakyReLu(256, 512, 3, 1, 1, bn=bn, b=spike_b)
        self.zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.down_sample6 = Conv_Bn_LeakyReLu(512, 512, 2, 1, 0, bn=False, b=spike_b)
        self.conv7 = Conv_Bn_LeakyReLu(512, 1024, 3, 1, 1, bn=bn, b=spike_b)
        self.conv8 = Conv_Bn_LeakyReLu(1024, 512, 3, 1, 1, bn=bn, b=spike_b)

        self.pred = nn.Conv2d(512, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1, bias=False)

    

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride[0], h // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def clear_mem(self):
        for m in self.modules():
            if isinstance(m, SpikeB):
                m.clear_mem()

    def forward(self, x):
        prediction = None
        self.clear_mem()
        B, T, C, H, W = x.shape
        x = x.view(B, -1, H, W)
        x1 = self.down_sample1(self.conv1(x))
        x2 = self.down_sample2(self.conv2(x1))
        x3 = self.down_sample3(self.conv3(x2))
        x4 = self.down_sample4(self.conv4(x3))
        x5 = self.down_sample5(self.conv5(x4))
        x6 = self.down_sample6(self.zero_pad(self.conv6(x5)))

        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        prediction = self.pred(x8)
        # for t in range(self.time_steps):
        #     x1 = self.down_sample1(self.conv1(x[:, t, ...]))
        #     x2 = self.down_sample2(self.conv2(x1))
        #     x3 = self.down_sample3(self.conv3(x2))
        #     x4 = self.down_sample4(self.conv4(x3))
        #     x5 = self.down_sample5(self.conv5(x4))
        #     x6 = self.down_sample6(self.zero_pad(self.conv6(x5)))

        #     x7 = self.conv7(x6)
        #     x8 = self.conv8(x7)
        #     pred = self.pred(x8)
        #     if prediction == None:
        #         prediction = torch.zeros_like(pred)
        #     prediction += pred
        
        # # 累加时间维度
        # prediction = prediction / self.time_steps

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W*num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous().view(B, H * W, self.num_anchors, 4)
        # [B, H*W*num_anchor, 4]
        # 该步将原本的坐标预测从相对于grid的(0,1)数据准换为相对于整个图片的(0,1)数据
        x1y1x2y2_pred = self.decode_bbox(txtytwth_pred) / self.input_size

        return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred

class YOLOv2Tiny_ANN_Speical(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, center_sample=False, bn=True, init_channels=5, time_steps=2, spike_b=3):
        super(YOLOv2Tiny_ANN_Speical, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = [32]
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        self.time_steps = time_steps


        self.conv1 = Conv_Bn_LeakyReLu(1, 16, 5, 2, 0, bn=bn, b=spike_b)    # (1, 251, 251)->(16, 124, 124)
        self.pool1 = nn.MaxPool2d(2, 2)                                     # (16, 124, 124)->(16, 62, 62)
        self.conv2 = Conv_Bn_LeakyReLu(16, 64, 3, 1, 1, bn=bn, b=spike_b)   # (16, 62, 62)->(64, 62, 62)
        self.pool1 = nn.MaxPool2d(2, 2, 1)                                  # (64, 62, 62)->(64, 32, 32)
        self.conv3 = Conv_Bn_LeakyReLu(64, 128, 3, 1, 1, bn=bn, b=spike_b)  # (64, 32, 32)->(128, 32, 32)
        self.conv4 = Conv_Bn_LeakyReLu(128, 128, 3, 1, 1, bn=bn, b=spike_b) # (128, 32, 32)->(128, 32, 32)
        self.conv5 = Conv_Bn_LeakyReLu(128, 128, 3, 1, 1, bn=bn, b=spike_b) # (128, 32, 32)->(128, 32, 32)
        self.pool3 = nn.MaxPool2d(2, 2)                                     # (128, 32, 32)->(128, 16, 16)
        self.conv6 = Conv_Bn_LeakyReLu(128, 256, 3, 1, 1, bn=bn, b=spike_b) # (128, 16, 16)->(256, 16, 16)
        self.conv7 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b) # (256, 16, 16)->(256, 16, 16)
        self.pool4 = nn.MaxPool2d(2, 2)                                     # (256, 16, 16)->(256, 8, 8)
        self.conv8 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b) # (256, 8, 8)->(256, 8, 8)
        self.pred = nn.Conv2d(256, 21, kernel_size=1, bias=False)

    

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride[0], h // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def clear_mem(self):
        for m in self.modules():
            if isinstance(m, SpikeB):
                m.clear_mem()

    def forward(self, x):
        prediction = None
        self.clear_mem()
        B, T, C, H, W = x.shape
        x = x.view(B, -1, H, W)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.down_sample1(self.conv3(x2))
        x4 = self.down_sample2(self.conv4(x3))
        x5 = self.down_sample3(self.zero_pad(self.conv5(x4)))

        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        prediction = self.pred(x7)

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W*num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:(1+self.num_classes+4) * self.num_anchors].contiguous().view(B, H * W, self.num_anchors, 4)
        # [B, H*W*num_anchor, 4]
        # 该步将原本的坐标预测从相对于grid的(0,1)数据准换为相对于整个图片的(0,1)数据
        x1y1x2y2_pred = self.decode_bbox(txtytwth_pred) / self.input_size

        return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred

class YOLOv2Tiny_ANN_Fit(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, center_sample=False, bn=True, init_channels=5, time_steps=2, spike_b=3):
        super(YOLOv2Tiny_ANN_Fit, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = [32]
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        self.time_steps = time_steps

        self.conv1 = Conv_Bn_LeakyReLu(init_channels * time_steps, 32, 5, 4, 2, bn=bn, b=spike_b)
        self.down_sample1 = Conv_Bn_LeakyReLu(32, 64, 3, 2, 1, bn=False, b=spike_b)
        self.conv2 = Conv_Bn_LeakyReLu(64, 128, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample2 = Conv_Bn_LeakyReLu(128, 128, 3, 2, 1, bn=False, b=spike_b)
        self.conv3 = Conv_Bn_LeakyReLu(128, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample3 = Conv_Bn_LeakyReLu(256, 256, 3, 2, 1, bn=False, b=spike_b)
        self.conv4 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.conv5 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.conv6 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)

        self.pred = nn.Conv2d(256, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1, bias=False)

    

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride[0], h // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def get_bias_parameters(self):
        parameters_list = []
        for name, param in self.named_parameters():
            if 'bias' in name:
                parameters_list.append(param)
        return parameters_list


    def get_exclude_bias_parameters(self):
        parameters_list = []
        for name, param in self.named_parameters():
            if 'bias' not in name:
                parameters_list.append(param)
        return parameters_list
    
    def cal_bias_loss(self):
        bias_loss = torch.zeros(1).to(self.device)
        for m in self.modules():
            if isinstance(m, Conv_Bn_LeakyReLu):
                conv = m.layer[0]
                if not isinstance(m.layer[1], nn.BatchNorm2d):
                    continue
                bn = m.layer[1]
                bias_loss += torch.mean((conv.bias-bn.running_mean) ** 2)
        return bias_loss

    def forward(self, x):
        prediction = None
        B, T, C, H, W = x.shape
        x = x.view(B, -1, H, W)
        x = self.down_sample1(self.conv1(x))
        x = self.down_sample2(self.conv2(x))
        x = self.down_sample3(self.conv3(x))
        x = self.conv4(x)

        x = self.conv5(x)
        x = self.conv6(x)
        prediction = self.pred(x)

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W*num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous().view(B, H * W, self.num_anchors, 4)
        # [B, H*W*num_anchor, 4]
        # 该步将原本的坐标预测从相对于grid的(0,1)数据准换为相对于整个图片的(0,1)数据
        x1y1x2y2_pred = self.decode_bbox(txtytwth_pred) / self.input_size

        # if self.trainable:
        #     return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, self.cal_bias_loss()
        # return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, self.cal_bias_loss()

        return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred



class YOLOv2Tiny_SNN_ANN_Fit(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, center_sample=False, bn=True, init_channels=5, time_steps=2, spike_b=3):
        super(YOLOv2Tiny_SNN_ANN_Fit, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = [32]
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        self.time_steps = time_steps

        self.conv1 = Conv_Bn_Spike(init_channels, 32, 5, 4, 2, bn=bn, b=spike_b)
        self.down_sample1 = Conv_Bn_LeakyReLu(32, 64, 3, 2, 1, bn=False, b=spike_b)
        self.conv2 = Conv_Bn_LeakyReLu(64, 128, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample2 = Conv_Bn_LeakyReLu(128, 128, 3, 2, 1, bn=False, b=spike_b)
        self.conv3 = Conv_Bn_LeakyReLu(128, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample3 = Conv_Bn_LeakyReLu(256, 256, 3, 2, 1, bn=False, b=spike_b)
        self.conv4 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.conv5 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.conv6 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)

        self.pred = nn.Conv2d(256, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1, bias=False)

    

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride[0], h // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred
    
    def get_decay(self):
        decay_list = []
        for m in self.modules():
            if isinstance(m, SpikeB):
                decay_list.append(m.decay.item())
        return decay_list

    def clear_mem(self):
        for m in self.modules():
            if isinstance(m, SpikeB):
                m.clear_mem()

    def forward(self, x):
        prediction = None
        self.clear_mem()
        for t in range(self.time_steps):
            y = self.down_sample1(self.conv1(x[:, t, ...]))
            y = self.down_sample2(self.conv2(y))
            y = self.down_sample3(self.conv3(y))
            y = self.conv4(y)
            y = self.conv5(y)
            y = self.conv6(y)
            pred = self.pred(y)
            if prediction == None:
                prediction = torch.zeros_like(pred)
            prediction += pred
        
        # 累加时间维度
        prediction = prediction / self.time_steps

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W*num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous().view(B, H * W, self.num_anchors, 4)
        # [B, H*W*num_anchor, 4]
        # 该步将原本的坐标预测从相对于grid的(0,1)数据准换为相对于整个图片的(0,1)数据
        x1y1x2y2_pred = self.decode_bbox(txtytwth_pred) / self.input_size

        return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred

class YOLOv2Tiny_LTC_ANN_Fit(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, center_sample=False, bn=True, init_channels=5, time_steps=2, spike_b=3):
        super(YOLOv2Tiny_LTC_ANN_Fit, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = [32]
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        self.time_steps = time_steps

        self.conv1 = Conv_Bn_LTC(init_channels, 32, 5, 4, 2, bn=bn, b=spike_b)
        self.down_sample1 = Conv_Bn_LeakyReLu(32, 64, 3, 2, 1, bn=False, b=spike_b)
        self.conv2 = Conv_Bn_LeakyReLu(64, 128, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample2 = Conv_Bn_LeakyReLu(128, 128, 3, 2, 1, bn=False, b=spike_b)
        self.conv3 = Conv_Bn_LeakyReLu(128, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample3 = Conv_Bn_LeakyReLu(256, 256, 3, 2, 1, bn=False, b=spike_b)
        self.conv4 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.conv5 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.conv6 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)

        self.pred = nn.Conv2d(256, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1, bias=False)

    

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride[0], h // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred
    
    def get_decay(self):
        decay_list = []
        for m in self.modules():
            if isinstance(m, LTC):
                decay_list.append(m.decay.item())
        return decay_list

    def clear_mem(self):
        for m in self.modules():
            if isinstance(m, LTC):
                m.clear_mem()

    def forward(self, x):
        prediction = None
        self.clear_mem()
        for t in range(self.time_steps):
            y = self.down_sample1(self.conv1(x[:, t, ...]))
            y = self.down_sample2(self.conv2(y))
            y = self.down_sample3(self.conv3(y))
            y = self.conv4(y)
            y = self.conv5(y)
            y = self.conv6(y)
            pred = self.pred(y)
            if prediction == None:
                prediction = torch.zeros_like(pred)
            prediction += pred
        
        # 累加时间维度
        prediction = prediction / self.time_steps

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W*num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous().view(B, H * W, self.num_anchors, 4)
        # [B, H*W*num_anchor, 4]
        # 该步将原本的坐标预测从相对于grid的(0,1)数据准换为相对于整个图片的(0,1)数据
        x1y1x2y2_pred = self.decode_bbox(txtytwth_pred) / self.input_size

        return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred


if __name__ == '__main__':
    net = YOLOv2Tiny_ANN_Fit(device=torch.device('cpu'), 
                   input_size=320, 
                   num_classes=2, 
                   trainable=True, 
                   anchor_size=anchor_size, 
                   center_sample=False,
                   time_steps=2,
                   spike_b=3,
                   bn=True,
                   init_channels=1)
    print(net)
