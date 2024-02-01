import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
from models.snndarts_search.SNN import*
from torch.nn.utils.fusion import fuse_conv_bn_eval
OPS = {
    'skip_connect': lambda Cin,Cout, stride, signal: Identity(Cin, Cout, signal) if stride == 1 else FactorizedReduce(Cin, Cout),
    # 'conv_3x3': lambda C, stride: ConvBR(C, C, 3, stride, 1),
    # 'conv_5x5': lambda C, stride: ConvBR(C, C, 5, stride, 2),
    'snn_b3': lambda Cin,Cout, stride, signal: SNN_2d(Cin, Cout, kernel_size=3, stride=stride,b=3),  # TODO change b
    'snn_b5': lambda Cin,Cout, stride, signal: SNN_2d(Cin, Cout, kernel_size=3, stride=stride,b=5)
}

class NaiveBN(nn.Module):
    def __init__(self, C_out, momentum=0.1):
        super(NaiveBN, self).__init__()
        self.op = nn.Sequential(
            nn.BatchNorm2d(C_out),
            nn.ReLU()
        )
        self._initialize_weights()

    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ConvBR(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bn=True, relu=True,dilation=1):
        super(ConvBR, self).__init__()
        self.relu = relu
        self.use_bn = bn

        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(C_out)
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

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                          bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=False)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Identity(nn.Module):
    def __init__(self, C_in, C_out, signal):
        super(Identity, self).__init__()
        self._initialize_weights()
        self.conv1 = nn.Conv2d(C_in,C_out,1,1,0)
        self.signal = signal

    def forward(self, x):
        if self.signal:
            return self.conv1(x)
        else:
            return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
    
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
        self._initialize_weights()

    def forward(self, x):
        # out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])], dim=1)
        out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DoubleFactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(DoubleFactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
        self._initialize_weights()

    def forward(self, x):
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class FactorizedIncrease(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DoubleFactorizedIncrease(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleFactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



# OPS = {
#     'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
#     'conv_3x3': lambda C, stride: ConvLTC_v1(input_c=C, output_c=C, kernel_size=3, stride=stride, padding=1)
# }



class ConvLTC_v1(nn.Module):
    '''more general discrete form of LTC'''  
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, ode_unfolds=1):
        super().__init__()
        hparams = {'use_erevin':False, 'num_plane':input_c, 'taum_ini':[0.5,.8], 'nltc': output_c, 'usetaum':True, 'use_ltc':True, 'use_relu':0, 'use_ltcsig':False, 'use_vtaum':False}
       # torch.manual_seed(0)
       # torch.cuda.manual_seed(0)
        # in_channels, num_features, tau_input, taum_ini, usetaum, stream_opt, self.burn_in_time = hparams['num_plane'], hparams['nltc'], hparams['use_erevin'], hparams['taum_ini'], hparams['usetaum'], hparams['stream_opt'], hparams['burn_in_time']
        in_channels, num_features, tau_input, taum_ini, usetaum = hparams['num_plane'], hparams['nltc'], hparams['use_erevin'], hparams['taum_ini'], hparams['usetaum']
        self.use_relu = hparams['use_relu']
        self.use_ltcsig = hparams['use_ltcsig']
        self.use_vtaum = hparams['use_vtaum']
        self.in_channels = in_channels
        self.num_features = num_features
        self.conv = self._make_layer(in_channels, num_features, kernel_size, padding, stride)
        self.usetaum = usetaum    
        # self.stream_opt = stream_opt
        self.cm = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)
        self.vleak = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)
        if self.usetaum:
            if self.use_vtaum:
                self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,num_features,1,1)+taum_ini[1])
            else:
                self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,1,1)+taum_ini[1])
            # self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,320,384)+taum_ini[1])
        else:
            self.gleak = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,1,1)+taum_ini[1])

        if self.use_ltcsig:
            self.mu = nn.Parameter(0.1*torch.randn(num_features,1,1)+0)
            self.sigma = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)

        #self.tau_m = nn.Parameter((1.-5.)*torch.rand(num_features,1,1)+5.)
        self.E_revin = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)# mean=1.0,std=0.1     

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU()

        #self._ode_unfolds = torch.Tensor(ode_unfolds).cuda()
        self._epsilon = 1e-8
        
        self.sigmoid = nn.Sigmoid()


        self.tau_input = tau_input
        self.tanh = nn.Tanh()
        self.debug = None
        self.debug1 = []
        self.debug2 = []
        self.debug3 = []
        self.counter = 0
        print("self.counter")

        nn.init.xavier_normal_(self.conv[0].weight.data)


    def ltc_sigmoid(self, v_pre, mu, sigma):
        mues = v_pre - mu
        x = sigma * mues
        return self.sigmoid(x)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def apply_weight_constraints(self):
    #        self.cm.data = self._clip(self.cm.data)
    #        self.gleak.data = self._clip(self.gleak.data)
        self.cm.data.clamp_(0,1000)
        self.vleak.data.clamp_(0,1000)
        if self.usetaum:
            self.tau_m.data.clamp_(0,2000)
            # self.tau_m.data.clamp_(0,1)
        else:
            self.gleak.data.clamp_(0,1000)
        # self.tau_m.data = self._clip(self.tau_m.data)
    
    def forward(self, inputs, v_pre=None):
        # self.counter+=1
        # print("!!!!!counter:",self.counter)
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
        if len(inputs.size()) == 4: # for bug in stream test
            B, C, H, W = inputs.size()
            # B= 1
            # inputs = torch.unsqueeze(inputs,0)
        else:
            B, C, S, H, W = inputs.size()
            # print('bcshw ', B,C,S,H,W)
        # if self.in_channels == 5:
            # S = S//5

        # v_pre = nn.Parameter(torch.zeros(B, self.num_features, H, W)).cuda()
        outputs = []
        # print("input.size()",inputs.size()) # 1 2 10 h w
       # cm_t = self.cm / (1. / self._ode_unfolds)
        cm_t = self.cm 
       # if is_train:
        #    cm_t.retain_grad()

        if type(v_pre) == int:
            v_pre = torch.zeros(B, self.num_features, H, W).cuda()
        else:
            v_pre = v_pre.to(inputs.device)
        # v_pre = torch.zeros(B, self.num_features, H, W).cuda()

        # v_pre.requires_grad  = False
        v_pre = v_pre.detach()

        # else:
            # print('bro, not bad')
        # S = 1            
        # for t in range(S-1,-1,-1):     
        # wih = self.conv(self.sigmoid(inputs[:, :,t])) # wi*sig(x)+wh*sig(vpre)
        # wih = self.conv(inputs[:, :,t]) # wi*sig(x)+wh*sig(vpre)
            # wih = self.conv(inputs[:, 0,int(t*5):int((t+1)*5)]) # wi*sig(x)+wh*sig(vpre)
            # wih = self.conv(inputs[:, 0,int(t*5):int((t+1)*5)]) # wi*sig(x)+wh*sig(vpre)
        wih = self.conv(inputs) # wi*sig(x)+wh*sig(vpre)
    

        if self.use_relu==1:
            wih = self.relu(wih)
        elif self.use_relu==2:
            wih = self.lrelu(wih)
        elif self.use_relu==3:
            wih = self.sigmoid(wih)
        # denominator = self.cm_t + self.gleak 
        # print('self.tau_m.shape',self.tau_m.shape)
        # for i in range(self.tau_m.shape[0]):
        #     if self.tau_m[i,0,0] < 0:
        #         self.tau_m[i,0,0]  = 0
        self.apply_weight_constraints()
        if self.tau_input:
            if self.usetaum:
                numerator = (
                    self.tau_m * v_pre / (self.vleak + self.cm*self.sigmoid(wih)) + wih*self.E_revin # ltcv3                      
                    # (self.tau_m + self.cm*(self.sigmoid(wih)-.5)) * v_pre + wih*self.E_revin  # ltcv4                     
                    # (self.tau_m * self.cm* self.sigmoid(wih)) * v_pre + wih*self.E_revin # ltcv5                       
                    # self.tau_m * v_pre / (self.vleak + self.sigmoid(wih)) + wih*self.E_revin                       
                )
                denominator = 1
                # print(S, t)
                # self.debug1.append(self.tau_m / (self.vleak + self.cm*self.sigmoid(wih)))
                # self.debug2.append(v_pre)
                # self.debug3.append(wih)
            # self.debugv_pre
            else:
                numerator = (
                cm_t * v_pre
                + self.gleak * self.vleak
                + wih*self.E_revin
                )
                denominator = cm_t + self.gleak + wih

        else:
            if self.usetaum:
                if self.use_vtaum:
                    numerator = (
                        torch.sum(self.tau_m * v_pre.unsqueeze(2),1) + wih# cc11*bc1hw cc*c1
                        # self.tau_m * (v_pre + wih)# *self.E_revin
                    )
                else:
                    # BUG RuntimeError: The size of tensor a (16) must match the size of tensor b (8) at non-singleton dimension 0
                    numerator = (self.tau_m * v_pre + wih)

                denominator = 1
                # denominator = 1 + self.tau_m
                # self.debug1.append(self.tau_m / (self.vleak + self.cm*self.sigmoid(wih)))
                # self.debug2.append(v_pre)
                # self.debug3.append(wih)
            else:
                numerator = (
                cm_t * v_pre
                + self.gleak * self.vleak
                + wih
                )
                denominator = cm_t + self.gleak


        v_pre = numerator / (denominator + self._epsilon) # [b c h w]

        # v_pre = self.tanh(v_pre)
        if self.use_ltcsig:
            v_pre = self.ltc_sigmoid(v_pre, self.mu, self.sigma)
        else:
            v_pre = self.sigmoid(v_pre)
     
        outputs.append(v_pre)

        self.debug = outputs[-1]
        # self.debug1 = torch.cat(self.debug1, 0)
        # self.debug2 = torch.cat(self.debug2, 0)
        # if self.stream_opt:   
        #     # return torch.cat(outputs, 0).reshape(S, -1, H, W)[self.burn_in_time:] # only work for B=1
        #     return torch.cat(outputs, 0).reshape(S, -1, H, W)[self.burn_in_time:] # only work for B=1
        # else:
        return outputs[-1]





class ConvLTC_inside_loop(nn.Module):
    '''more general discrete form of LTC'''  
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, ode_unfolds=1):
        super().__init__()
       # torch.manual_seed(0)
       # torch.cuda.manual_seed(0)
        hparams={'use_erevin':False, 'num_plane':input_c, 'taum_ini':[.5,.8], 'nltc': output_c, 'usetaum':True, 'ltcv1':True, 'stream_opt':False,'burn_in_time':5}
        in_channels, num_features, tau_input, taum_ini, usetaum, stream_opt, self.burn_in_time = hparams['num_plane'], hparams['nltc'], hparams['use_erevin'], hparams['taum_ini'], hparams['usetaum'], hparams['stream_opt'], hparams['burn_in_time']
        self.in_channels = in_channels
        self.num_features = num_features
        self.conv = self._make_layer(in_channels, num_features, kernel_size, padding, stride)
        self.usetaum = usetaum    
        self.stream_opt = stream_opt
        self.cm = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)
        self.vleak = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)
        if self.usetaum:
            self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,1,1)+taum_ini[1])
        else:
            self.gleak = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,1,1)+taum_ini[1])

        #self.tau_m = nn.Parameter((1.-5.)*torch.rand(num_features,1,1)+5.)
        self.E_revin = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)# mean=1.0,std=0.1     
        
        #self._ode_unfolds = torch.Tensor(ode_unfolds).cuda()
        self._epsilon = 1e-8
        
        self.sigmoid = nn.Sigmoid()
        self.tau_input = tau_input
        self.tanh = nn.Tanh()
        self.debug = None

        nn.init.xavier_normal_(self.conv[0].weight.data)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    #def init_v_pre(self, B, num_features, H, W):
    #    self.v_pre = nn.Parameter(torch.zeros(B, self.num_features, H, W)).cuda()

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
    
    def forward(self, inputs, is_train=True):
        # print('ltc input',inputs.shape)
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
        inputs = rearrange(inputs, 'b s c h w -> b c s h w')
        B, C, S, H, W = inputs.size()
        # if self.in_channels == 5:
        #     S = S//5

        # v_pre = nn.Parameter(torch.zeros(B, self.num_features, H, W)).cuda()
        outputs = []
        # print("input.size()",inputs.size()) # 1 2 10 h w
       # cm_t = self.cm / (1. / self._ode_unfolds)
        cm_t = self.cm 
       # if is_train:
        #    cm_t.retain_grad()
        v_pre = torch.zeros(B, self.num_features, H, W).cuda()
        for t in range(S-1,-1,-1):     
            # print(counter)
            # counter+=1
            # wih = self.conv(self.sigmoid(inputs[:, :,t])) # wi*sig(x)+wh*sig(vpre)
            # wih = self.conv(inputs[:, :,t]) # wi*sig(x)+wh*sig(vpre)
            # if self.in_channels == 5:
            #     wih = self.conv(inputs[:, 0,int(t*5):int((t+1)*5)]) # wi*sig(x)+wh*sig(vpre)
            # else:
            wih = self.conv(inputs[:, :,t]) # wi*sig(x)+wh*sig(vpre)

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
            return torch.cat(outputs, 0).reshape(S, -1, H, W)[self.burn_in_time:] # only work for B=1
        else:
            return outputs[-1]