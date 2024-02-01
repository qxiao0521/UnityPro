from turtle import left
import torch.nn.functional as F
from models.operations_2d import *
from models.genotypes_2d import PRIMITIVES
import numpy as np

class eca_block(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)

        return y
        # return x * y.expand_as(x)

class attention_shuffle(nn.Module):
    def __init__(self,channel, k_size=3):
        super(attention_shuffle,self).__init__()
        self.eca_block = eca_block(channel)

    def forward(self,x):
        weight = self.eca_block(x).squeeze()
        # print(weight.size())
        switch = 0
        if switch == 0:# return 4 highest
            _, indices = torch.sort(weight.squeeze(), dim=1,descending=True)

        elif switch == 1:# return 4 lowest
            _, indices = torch.sort(weight.squeeze(), dim=1,descending=False)

        elif switch == 2:# return 2 highest and 2 lowest
            _, indices = torch.sort(weight.squeeze())
            indices[3],indices[4] = indices[-2], indices[-1]

        y = torch.stack([x[i,indices[i]] for i in range(x.size()[0])])
        return y

def random_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    indices = torch.randperm(num_channels)
    x = x[:,indices]
    return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):
    def __init__(self, C_in, C, stride, p,signal):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.p = p
        self.k = 4
        self.partial_channel = 0
        self.C_in = C_in
        self.C = C
        if self.partial_channel == 1 and C_in == C:
            # self.attention_shuffle = attention_shuffle(C)
            for primitive in PRIMITIVES:
                op = OPS[primitive](C_in//self.k,C//self.k, stride,signal)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C))
                if isinstance(op,Identity) and p>0:
                    op = nn.Sequential(op, nn.Dropout(self.p))
                self._ops.append(op)
        
        else:
            for primitive in PRIMITIVES:
                op = OPS[primitive](C_in, C, stride,signal)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C))
                if isinstance(op,Identity) and p>0:
                    op = nn.Sequential(op, nn.Dropout(self.p))
                self._ops.append(op)



    def update_p(self):
        for op in self._ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p
    

    def forward(self, x, weights, left_or_right):
        # print(type(x),x.shape)
        if self.partial_channel == 1:
            x = self.attention_shuffle(x)
            # print(x.size())
            dim_2 = x.shape[1]
            xtemp = x[ : , :  dim_2//self.k, :, :]
            xtemp2 = x[ : ,  dim_2//self.k:, :, :]
            temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
            #reduction cell needs pooling before concat
            # if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1,xtemp2],dim=1)
            # else:
            #   ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
                # ans = channel_shuffle(ans,self.k)
                #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
                #except channe shuffle, channel shift also works
            return ans 
            
        else:
            opt_outs = []
            for i in range(3):
                if i == 0:
                    opt_out = self._ops[i](x)
                    opt_out = weights[i] * opt_out
                    opt_outs.append(opt_out)
                else:
                    opt_out = self._ops[i](x, left_or_right)
                    opt_out = weights[i] * opt_out
                    opt_outs.append(opt_out)
            return sum(opt_outs)


class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_fmultiplier_down, prev_fmultiplier_same, prev_fmultiplier_up,
                 filter_multiplier,p=0.0):

        super(Cell, self).__init__()

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier

        self.p = p
        self.C_prev_prev = int(prev_prev_fmultiplier * block_multiplier)
        self._prev_fmultiplier_same = prev_fmultiplier_same

        # if prev_fmultiplier_down is not None:
        #     self.C_prev_down = int(prev_fmultiplier_down * block_multiplier)
        #     self.preprocess_down = ConvBR(
        #         self.C_prev_down, self.C_out, 1, 1, 0)
        # if prev_fmultiplier_same is not None:
        #     self.C_prev_same = int(prev_fmultiplier_same * block_multiplier)
        #     self.preprocess_same = ConvBR(
        #         self.C_prev_same, self.C_out, 1, 1, 0)
        # if prev_fmultiplier_up is not None:
        #     self.C_prev_up = int(prev_fmultiplier_up * block_multiplier)
        #     self.preprocess_up = ConvBR(
        #         self.C_prev_up, self.C_out, 1, 1, 0)

        # if prev_prev_fmultiplier != -1:
        #     self.pre_preprocess = ConvBR(
        #         self.C_prev_prev, self.C_out, 1, 1, 0)

        self._steps = steps
        self.block_multiplier = block_multiplier
        # self._ops = nn.ModuleList()
        self._ops_down = nn.ModuleList()
        self._ops_same = nn.ModuleList()
        self._ops_up = nn.ModuleList()

        # self.left_vpre = [None]*20
        # self.left_vpre = np.array(self.left_vpre).reshape(10,2)
        # self.left_vpre = list(self.left_vpre)
        # self.right_vpre = [None]*20
        # self.right_vpre = np.array(self.right_vpre).reshape(10,2)
        # self.right_vpre = list(self.right_vpre)

        # for i in range(self._steps):
        #     for j in range(2 + i):
        #         stride = 1
        #         if prev_prev_fmultiplier == -1 and j == 0:
        #             op = None
        #             # l_v = None
        #         else:
        #             op = MixedOp(self.C_out, stride, self.p)
        #         self._ops.append(op)
                # self.left_vpre.append(l_v)
                # self.right_vpre.append(l_v)

        if prev_fmultiplier_down is not None:
            c_prev_down = int(prev_fmultiplier_down * block_multiplier)
            for i in range(self._steps):
                for j in range(2 + i):
                    stride = 2
                    if prev_prev_fmultiplier == -1 and j == 0:
                        op = None
                    else:
                        if j == 0:
                            op = MixedOp(self.C_prev_prev,self.C_out, 1,self.p,signal=1)
                        elif j == 1:
                            op = MixedOp(c_prev_down, self.C_out, 2, self.p,signal=1)
                        else:
                            op = MixedOp(self.C_out, self.C_out, 1, self.p,signal=0)
                    self._ops_down.append(op)



        if prev_fmultiplier_same is not None:
            c_prev_same = int(prev_fmultiplier_same * block_multiplier)
            for i in range(self._steps):
                for j in range(2 + i):
                    stride = 1
                    if prev_prev_fmultiplier == -1 and j == 0:
                        op = None
                    else:
                        if j == 0:
                            op = MixedOp(self.C_prev_prev,self.C_out, stride,self.p,signal=1)
                        elif j == 1:
                            op = MixedOp(c_prev_same, self.C_out, stride, self.p,signal=1)
                        else:
                            op = MixedOp(self.C_out, self.C_out, stride, self.p,signal=0)
                    self._ops_same.append(op)


        if prev_fmultiplier_up is not None:
            c_prev_up = int(prev_fmultiplier_up * block_multiplier)
            for i in range(self._steps):
                for j in range(2 + i):
                    stride = 1
                    if prev_prev_fmultiplier == -1 and j == 0:
                        op = None
                    else:
                        if j == 0:
                            op = MixedOp(self.C_prev_prev,self.C_out, stride,self.p,signal=1)
                        elif j == 1:
                            op = MixedOp(c_prev_up, self.C_out, stride, self.p,signal=1)
                        else:
                            op = MixedOp(self.C_out, self.C_out, stride, self.p,signal=0)
                    self._ops_up.append(op)


        self._initialize_weights()

    def update_p(self):
        for op in self._ops_down:
            if op == None:
                continue
            else:
                op.p = self.p
                op.update_p()
        for op in self._ops_same:
            if op == None:
                continue
            else:
                op.p = self.p
                op.update_p()
        for op in self._ops_up:
            if op == None:
                continue
            else:
                op.p = self.p
                op.update_p()

    def scale_dimension(self, dim, scale):
        assert isinstance(dim, int)
        # return int(dim*scale)
        #if dim % 2 == 0:
        #    return int(dim * scale)
        #else:
        return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 else int(dim * scale)

    def prev_feature_resize(self, prev_feature, mode):
        if mode == 'down':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 0.5)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 0.5)
        elif mode == 'up':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 2)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 2)

        return F.interpolate(prev_feature, (feature_size_h, feature_size_w), mode='bilinear', align_corners=True)

    def forward(self, s0, s1_down, s1_same, s1_up, n_alphas, left_or_right):
        # if left_or_right == 2 or left_or_right == 3:
        #     self.left_vpre = [None]*20
        #     self.left_vpre = np.array(self.left_vpre).reshape(10,2)
        #     self.left_vpre = list(self.left_vpre)
        #     self.right_vpre = [None]*20 # TODO find better code way
        #     self.right_vpre = np.array(self.right_vpre).reshape(10,2)
        #     self.right_vpre = list(self.right_vpre)
        #     left_or_right -=2

        # use s1_same's size to resize s1_up
        if s1_same is not None: # 12,68,94
            size_h, size_w = s1_same.shape[2], s1_same.shape[3] 
        if s1_up is not None: # 24,34,47->24,68,94->24,68,94
            s1_up = self.prev_feature_resize(s1_up, 'up')
            if s1_up.shape[2] != size_h or s1_up.shape[3] != size_w:
                s1_up = F.interpolate(s1_up, (size_h, size_w),mode='bilinear',align_corners=True)

        # DEBUG
        # print('#####cell')
        # if s1_down is not None:
        #     print('s1_down',s1_down.shape)
        # if s1_same is not None:
        #     print('s1_same',s1_same.shape)
        # if s1_up is not None:
        #     print('s1_up',s1_up.shape)

        final_concates = []
        if s1_down is not None:
            states = [s0, s1_down]
            offset = 0
            for i in range(self._steps):
                new_states = []
                for j, h in enumerate(states):
                    branch_index = offset + j
                    if self._ops_down[branch_index] is None:
                        continue             
                    new_state = self._ops_down[branch_index](h, n_alphas[branch_index], left_or_right)
                    new_states.append(new_state)
                s = sum(new_states)
                offset += len(states)
                states.append(s)
            concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
            # print(concat_feature.shape)
            
            final_concates.append(concat_feature)


        if s1_same is not None:
            states = [s0, s1_same]
            offset = 0
            for i in range(self._steps):
                new_states = []
                for j, h in enumerate(states):
                    branch_index = offset + j
                    if self._ops_same[branch_index] is None:
                        continue              
                    new_state = self._ops_same[branch_index](h, n_alphas[branch_index], left_or_right)
                    new_states.append(new_state)
                s = sum(new_states)
                offset += len(states)
                states.append(s)
            concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
            final_concates.append(concat_feature)
            # a=b


        if s1_up is not None:
            states = [s0, s1_up]
            offset = 0
            for i in range(self._steps):
                # counter = 0
                new_states = []
                for j, h in enumerate(states):
                    branch_index = offset + j
                    if self._ops_up[branch_index] is None:
                        continue              
                    new_state = self._ops_up[branch_index](h, n_alphas[branch_index], left_or_right)
                    new_states.append(new_state)
                s = sum(new_states)
                offset += len(states)
                states.append(s)
            concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
            final_concates.append(concat_feature)
     
        return final_concates

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
