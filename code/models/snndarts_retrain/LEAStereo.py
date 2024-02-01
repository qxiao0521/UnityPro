import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.build_model_2d import Disp
from models.decoding_formulas import network_layer_to_space
from retrain.new_model_2d import newFeature
from retrain.skip_model_3d import newMatching
from matching import Matching
from matching import MatchingOperation
from models.SNN import SCNN_frontend
import time


class LEAStereo(nn.Module):
    def __init__(self, args):
        super(LEAStereo, self).__init__()

        network_path_fea, cell_arch_fea = np.load(args.net_arch_fea), np.load(args.cell_arch_fea)
        network_path_mat, cell_arch_mat = np.load(args.net_arch_mat), np.load(args.cell_arch_mat)
        print('Feature network path:{}\nMatching network path:{} \n'.format(network_path_fea, network_path_mat))

        network_arch_fea = network_layer_to_space(network_path_fea)
        network_arch_mat = network_layer_to_space(network_path_mat)

        self.maxdisp = 33
        # self.maxdisp = args.maxdisp
        self.feature = newFeature(5,network_arch_fea, cell_arch_fea, args=args)
        self.matching= newMatching(network_arch_mat, cell_arch_mat, args=args) 
        self.disp = Disp(self.maxdisp)
        # self.concatinate = Matching(args.maxdisp//3, MatchingOperation(24, 32, 24,2))
        self.snn_frontend = SCNN_frontend(input_c=5, output_c=16)
        #self._initialize_alphas()

    def forward(self, x, y, param): 
        param['mixed_at_mem'] = False
        #print('is_diffb',is_diffb)
        is_diffb = False
        img_device = torch.device('cuda', x.get_device())

        # x: B, 3, 10, 260, 346
        use_snn_frontend = False
        use_snn = True
        if use_snn_frontend:
            x = self.snn_frontend(x)
            y = self.snn_frontend(y)
        if use_snn:
            # 0 left, 1 right, 2 left init, 3 right init
            x_outs = []
            y_outs = []
            cost_all = []
            for i in range(x.shape[1]): # B,15,5,260,346 # TODO preframe = 10 # TODO batch = 10
                if i == 0:
                    param['is_first'] = True
                else:
                    param['is_first'] = False
                
                param['left_or_right'] = 'left'
                x_out = self.feature(x[:,i], param)

                param['left_or_right'] = 'right'
                y_out = self.feature(y[:,i], param)
                # print('y_out',y_out.shape)
                with torch.cuda.device_of(x_out):
                    # matching_signature = self.concatinate(x_out, y_out) # 4, 24, 22, 88, 116
                    # print('matching_signature',matching_signature.shape)
                    matching_signature = x_out.new().resize_(x_out.size()[0], x_out.size()[1]*2, self.maxdisp,  x_out.size()[2],  x_out.size()[3]).zero_()
                    for  j in range(self.maxdisp):
                        if j > 0 :
                            matching_signature[:,:x_out.size()[1], j,:,j:] = x_out[:,:,:,j:]
                            matching_signature[:,x_out.size()[1]:, j,:,j:] = y_out[:,:,:,:-j]
                        else:
                            matching_signature[:,:x_out.size()[1],j,:,j:] = x_out
                            matching_signature[:,x_out.size()[1]:,j,:,j:] = y_out
                cost = self.matching(matching_signature, param).squeeze(1)
                cost_all.append(cost.unsqueeze(1))
            cost = torch.cat(cost_all,1) # B, 15, 22, 88, 116
            cost = cost[:,2:]
            h = cost.shape[-2]
            w = cost.shape[-1]
            cost = cost.reshape(-1,self.maxdisp,h,w)

        if h == 88:
            cost = F.interpolate(cost,(260,346), mode='bilinear',align_corners=True) # 4, 22, 260, 346
        else:
            cost = F.interpolate(cost,(200,280), mode='bilinear',align_corners=True) # 4, 22, 260, 346

        return cost

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
                         + list(self.decoder.parameters()) \
                         + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params

    # def _initialize_alphas(self):
    #     # k = sum(1 for i in range(self._step) for n in range(2 + i))

    #     alphas = (1e-3 * torch.randn(10,6,3)).clone().detach().requires_grad_(True)
    #     #betas = (1e-3 * torch.randn(self._num_layers, 4, 3)).clone().detach().requires_grad_(True)

    #     self._arch_parameters = [
    #         alphas,
    #     ]
    #     self._arch_param_names = [
    #         'alphas',
    #     ]

    #     [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if 'alpha_diffb' in name]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'alpha_diffb' not in name]