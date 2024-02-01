import torch
import torch.nn as nn
import numpy as np
from models.genotypes_3d import PRIMITIVES
from models.genotypes_3d import Genotype
from models.operations_3d import *
import torch.nn.functional as F
import numpy as np
import pdb

class Cell(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, network_arch,
                 filter_multiplier, downup_sample, args=None):
        super(Cell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        # self.pre_preprocess = ConvBR(self.C_prev_prev, self.C_out, 1, 1, 0)
        # self.preprocess = ConvBR(self.C_prev, self.C_out, 1, 1, 0)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        # for x in self.cell_arch:
        #     primitive = PRIMITIVES[x[1]]
        #     op = OPS[primitive](self.C_out, stride=1)
        #     self._ops.append(op)
        self.cell_arch = torch.sort(self.cell_arch,dim=0)[0].to(torch.uint8)

        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            if x[0] in [0,2,5]:
                # op = OPS[primitive](self.C_prev_prev, self.C_out, stride=1)
                op = OPS[primitive](self.C_prev_prev, self.C_out, stride=1, signal=1)
            elif x[0] in [1,3,6]:
                op = OPS[primitive](self.C_prev, self.C_out, stride=1, signal=1)
            else:
                op = OPS[primitive](self.C_out, self.C_out, stride=1, signal=1)
            # primitive = PRIMITIVES[x[1]]
            # op = OPS[primitive](self.C_out, stride=1)
            self._ops.append(op) 
        self.mem = None
        self.act_fun = ActFun_changeable().apply

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input, param):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_d = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_h = self.scale_dimension(s1.shape[3], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[4], self.scale)
            s1 = F.interpolate(s1, [feature_size_d, feature_size_h, feature_size_w], mode='nearest')
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]) or (s0.shape[4] != s1.shape[4]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3], s1.shape[4]),
                                            mode='nearest')
        # s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
        # s1 = self.preprocess(s1)
        device = prev_input.device
        
        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    # new_state = self._ops[ops_index](h)
                    if isinstance(self._ops[ops_index],Identity):
                        new_state = self._ops[ops_index](h)
                    else:
                        param['mixed_at_mem'] = True
                        new_state = self._ops[ops_index](h, param)
                        if self.mem == None:
                            self.mem = [torch.zeros_like(new_state,device=device)]*self._steps
                        param['mixed_at_mem'] = False
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            self.mem[i] = self.mem[i] + s
            spike = self.act_fun(self.mem[i],3)
            self.mem[i] = self.mem[i] * decay * (1. - spike) 

            offset += len(states)
            states.append(spike)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1) 
        return prev_input, concat_feature

class newMatching(nn.Module):
    def __init__(self, network_arch, cell_arch, cell=Cell, args=None):        
        super(newMatching, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = args.mat_step
        self._num_layers = args.mat_num_layers
        self._block_multiplier = args.mat_block_multiplier
        self._filter_multiplier = args.mat_filter_multiplier
        
        initial_fm = self._filter_multiplier * self._block_multiplier
        half_initial_fm = initial_fm // 2

        self.stem0 = SNN_3d(initial_fm*2, initial_fm, kernel_size=3, stride=1, padding=1, b=3)
        self.stem1 = SNN_3d(initial_fm, initial_fm, kernel_size=3, stride=1, padding=1, b=3)
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()
            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             initial_fm / self._block_multiplier,
                             self.cell_arch, self.network_arch[i],
                             self._filter_multiplier * filter_param_dict[level],
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 initial_fm / self._block_multiplier,
                                 self._filter_multiplier * filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier * filter_param_dict[level],
                                 downup_sample, self.args)
                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier * filter_param_dict[prev_prev_level],
                                 self._filter_multiplier *
                                 filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier * filter_param_dict[level], downup_sample, self.args)

            self.cells += [_cell]

        self.last_3  = ConvBR(initial_fm, 1, 3, 1, 1,  bn=False, relu=False)  
        self.last_6  = SNN_3d(initial_fm*2 , initial_fm,    1, 1, 0)  
        # self.last_12 = ConvBR(initial_fm*4 , initial_fm*2,  1, 1, 0)  
        # self.last_24 = ConvBR(initial_fm*8 , initial_fm*4,  1, 1, 0)  
        
        # self.conv1 = ConvBR(initial_fm*4, initial_fm*2, 3, 1, 1)
        # self.conv2 = ConvBR(initial_fm*4, initial_fm*2, 3, 1, 1)

    def forward(self, x, param):
        # print(param)
        # if normalized_alphas is not None:
        if False:
        # print('feature l_r:', left_or_right)
            stem0 = self.stem0(x, left_or_right, B[4][0]-0.2)*normalized_alphas[4][0][0]+\
                    self.stem0(x, left_or_right, B[4][0])*normalized_alphas[4][0][1]+\
                    self.stem0(x, left_or_right, B[4][0]+0.2)*normalized_alphas[4][0][2]
            stem1 = self.stem1(stem0, left_or_right, B[5][0]-0.2)*normalized_alphas[5][0][0]+\
                    self.stem1(stem0, left_or_right, B[5][0])*normalized_alphas[5][0][1]+\
                    self.stem1(stem0, left_or_right, B[5][0]+0.2)*normalized_alphas[5][0][2]
            out = (stem0, stem1)
            out0 = self.cells[0](out[0], out[1], left_or_right, B[6], normalized_alphas[6])
            # print("out0",out0[0].size(), out0[1].size())
            out1 = self.cells[1](out0[0], out0[1], left_or_right, B[7], normalized_alphas[7])
            # print("out1",out1[0].size(), out1[1].size())
            out2 = self.cells[2](out1[0], out1[1], left_or_right, B[8], normalized_alphas[8])
            # print("out2",out2[0].size(), out2[1].size())
            out3 = self.cells[3](out2[0], out2[1], left_or_right, B[9], normalized_alphas[9])
        else:
            stem0 = self.stem0(x, param)
            stem1 = self.stem1(stem0, param)
            out = (stem0, stem1)
            out0 = self.cells[0](out[0], out[1], param)
            out1 = self.cells[1](out0[0], out0[1], param)
            out2 = self.cells[2](out1[0], out1[1], param)
            out3 = self.cells[3](out2[0], out2[1], param)


        # print("out3",out3[0].size(), out3[1].size())
        # out4 = self.cells[4](out3[0], out3[1], left_or_right, self.mid_b)
        residual_connection = False
        if residual_connection:
            if out3[-1].size(3) != out0[-1].size(3) or out3[-1].size(4) != out0[-1].size(4):
                d = out0[-1].size(2)
                h = out0[-1].size(3)
                w = out0[-1].size(4)

                temp_x = out3[0]
                temp_y = out3[-1]
                temp_y = F.interpolate(temp_y,(d, h, w), mode='trilinear', align_corners=True)
                out3 = (temp_x, temp_y)

                # print(out4[-1].size())
            # print("cat:",torch.cat((out1[-1], out4[-1]), 1).size()) 
            print('out0[-1], out3[-1]',out0[-1].shape, out3[-1].shape)
            out3_cat = self.conv1(torch.cat((out0[-1], out3[-1]), 1)) 
            last_output = out3_cat
        else:
            last_output = out3[-1]

        # out5 = self.cells[5](out4[0], out4_cat)
        # out5 = self.cells[5](out4[0],out4[1])
        # print("out5",out5[0].size(), out5[1].size())
        # out6 = self.cells[6](out5[0], out5[1])
        # print("out6",out6[0].size(), out6[1].size())
        # out7 = self.cells[7](out6[0], out6[1])
        
        
        # if out7[-1].size(3) != out4[-1].size(3) or out7[-1].size(4) != out4[-1].size(4):
        #     d = out4[-1].size(2)
        #     h = out4[-1].size(3)
        #     w = out4[-1].size(4)

        #     temp_x = out7[0]
        #     temp_y = out7[-1]
        #     temp_y = F.interpolate(temp_y,(d, h, w), mode='trilinear', align_corners=True)
        #     out7 = (temp_x, temp_y)
        # 
        # out7_cat = self.conv2(torch.cat((out4[-1], out7[-1]), 1))
        
        # print("out7_cat",out7[0].size(), out7[1].size())
        # out8 = self.cells[8](out7[0], out7_cat)
        # print("out8",out8[0].size(), out8[1].size())
        
        # print("out4,out8",out4[-1].size(),out8[-1].size())
        # out8_cat = self.conv2(torch.cat((out4[-1], out8[-1]), 1))
        # out9 = self.cells[9](out8[0],out8[1])
        # out9 = self.cells[9](out8[0], out8_cat)
        # out10= self.cells[10](out9[0], out9[1])
        # out11= self.cells[11](out10[0],out10[1])
        # last_output = out11[-1]
        

        d, h, w = x.size()[2], x.size()[3], x.size()[4]
        upsample_6  = nn.Upsample(size=x.size()[2:], mode='trilinear', align_corners=True)
        upsample_12 = nn.Upsample(size=[d//2, h//2, w//2], mode='trilinear', align_corners=True)
        upsample_24 = nn.Upsample(size=[d//4, h//4, w//4], mode='trilinear', align_corners=True)

        # print("3D last_output.size(), h",last_output.size(),h)
        if last_output.size()[3] == h:
            mat = self.last_3(last_output)
        elif last_output.size()[3] == h//2:
            mat = self.last_3(upsample_6(self.last_6(last_output, param)))
        elif last_output.size()[3] == 33:
            mat = self.last_3(upsample_6(self.last_6(last_output, param)))
        elif last_output.size()[3] == h//4:
            mat = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(last_output)))))
        elif last_output.size()[3] == 21 or last_output.size()[3] == 17:
            mat = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(last_output)))))
        elif last_output.size()[3] == h//8:
            mat = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(upsample_24(self.last_24(last_output)))))))      
        elif last_output.size()[3] == 9:
            mat = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(upsample_24(self.last_24(last_output)))))))     
        return mat  

