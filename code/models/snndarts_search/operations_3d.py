import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SNN import*
from torch.nn.utils.fusion import fuse_conv_bn_eval
OPS = {
    'skip_connect': lambda Cin, Cout, stride, signal: Identity(Cin, Cout, signal) if stride == 1 else FactorizedReduce(Cin, Cout),
    'snn_b3': lambda Cin, Cout, stride, signal: SNN_3d(Cin, Cout, kernel_size=3, stride=stride,b=3),
    'snn_b5': lambda Cin, Cout, stride, signal: SNN_3d(Cin, Cout, kernel_size=3, stride=stride,b=5)
}

class NaiveBN(nn.Module):
    def __init__(self, C_out, momentum=0.1):
        super(NaiveBN, self).__init__()
        self.op = nn.Sequential(
            nn.BatchNorm3d(C_out),
            nn.ReLU()
        )
        self._initialize_weights()

    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ConvBR(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bn=True, relu=True):
        super(ConvBR, self).__init__()
        self.relu = relu
        self.use_bn = bn

        self.conv = nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(C_out)
        # self._initialize_weights()

    def forward(self, x):
        if self.use_bn:
            if not self.bn.training:
                conv_bn = fuse_conv_bn_eval(self.conv,self.bn)
                x = conv_bn(x)
            else:
                x = self.bn(self.conv(x))
        else:
            x = self.conv(x)
        return x
        
class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                          bias=False),
            nn.Conv3d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_in),
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_out),
            nn.ReLU(inplace=False)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Identity(nn.Module):
    def __init__(self,C_in,C_out,signal):
        super(Identity, self).__init__()
        self._initialize_weights()
        self.conv1 = nn.Conv3d(C_in,C_out,1,1,0)
        self.signal = signal

    def forward(self, x):
        if self.signal:
            return self.conv1(x)
        else:
            return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv3d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(C_out)
        self._initialize_weights()

    def forward(self, x):
        # out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])], dim=1)
        out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv3d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DoubleFactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(DoubleFactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
        self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(C_out)
        self._initialize_weights()

    def forward(self, x):
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class FactorizedIncrease(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.Conv3d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=False)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DoubleFactorizedIncrease(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleFactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.Conv3d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.Conv3d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=False)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


