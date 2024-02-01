import torch
import torch.nn as nn

class ConvLTC_v1(nn.Module):

    '''more general discrete form of LTC'''
    def __init__(self, hparams, kernel_size=3, stride=1, padding=1, ode_unfolds=1):
        super().__init__()
       # torch.manual_seed(0)
       # torch.cuda.manual_seed(0)
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

    def forward(self, inputs, is_train):
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
        B, C, S, H, W = inputs.size()
        if self.in_channels == 5:
            S = S//5

        # v_pre = nn.Parameter(torch.zeros(B, self.num_features, H, W)).cuda()
        outputs = []
        # print("input.size()",inputs.size()) # 1 2 10 h w
       # cm_t = self.cm / (1. / self._ode_unfolds)
        cm_t = self.cm
       # if is_train:
        #    cm_t.retain_grad()
        v_pre = torch.zeros(B, self.num_features, H, W).cuda()
        for t in range(S-1,-1,-1):

            # wih = self.conv(self.sigmoid(inputs[:, :,t])) # wi*sig(x)+wh*sig(vpre)
            # wih = self.conv(inputs[:, :,t]) # wi*sig(x)+wh*sig(vpre)
            if self.in_channels == 5:
                wih = self.conv(inputs[:, 0,int(t*5):int((t+1)*5)]) # wi*sig(x)+wh*sig(vpre)
            else:
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



class ConvLTC_NAS(nn.Module):

    '''more general discrete form of LTC'''
    def __init__(self, hparams, kernel_size=3, stride=1, padding=1, ode_unfolds=1):
        super().__init__()
       # torch.manual_seed(0)
       # torch.cuda.manual_seed(0)
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
        self.counter = 0

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

    def forward(self, inputs, is_train):

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
        B, C, S, H, W = inputs.size()
        if self.in_channels == 5:
            S = S//5

        # v_pre = nn.Parameter(torch.zeros(B, self.num_features, H, W)).cuda()
        outputs = []
        # print("input.size()",inputs.size()) # 1 2 10 h w
       # cm_t = self.cm / (1. / self._ode_unfolds)
        cm_t = self.cm
       # if is_train:
        #    cm_t.retain_grad()
        v_pre = torch.zeros(B, self.num_features, H, W).cuda()
        for t in range(S-1,-1,-1):

            # wih = self.conv(self.sigmoid(inputs[:, :,t])) # wi*sig(x)+wh*sig(vpre)
            # wih = self.conv(inputs[:, :,t]) # wi*sig(x)+wh*sig(vpre)
            if self.in_channels == 5:
                wih = self.conv(inputs[:, 0,int(t*5):int((t+1)*5)]) # wi*sig(x)+wh*sig(vpre)
            else:
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