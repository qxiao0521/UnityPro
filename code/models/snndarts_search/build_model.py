import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build_model_2d import AutoFeature, Disp
from models.build_model_3d import AutoMatching
import pdb
from time import time
from matching import Matching
from matching import MatchingOperation
import fitlog
from models.operations_2d import ConvLTC_inside_loop
from models.SNN import SCNN_frontend
class AutoStereo(nn.Module):
    def __init__(self, frame_rate, maxdisp=192, Fea_Layers=6, Fea_Filter=8, Fea_Block=4, Fea_Step=3, Mat_Layers=12, Mat_Filter=8, Mat_Block=4, Mat_Step=3, p=0.0):
        super(AutoStereo, self).__init__()
        self.maxdisp = maxdisp
        #define Feature parameters
        self.Fea_Layers = Fea_Layers # 6
        self.Fea_Filter = Fea_Filter  # 8
        self.Fea_Block = Fea_Block # 4
        self.Fea_Step = Fea_Step # 3
        #define Matching parameters
        self.Mat_Layers = Mat_Layers
        self.Mat_Filter = Mat_Filter
        self.Mat_Block = Mat_Block
        self.Mat_Step = Mat_Step
        self.concatinate = Matching(maxdisp//3,MatchingOperation(36, 54, 36, 2))

        # self.feature  = AutoFeature(frame_rate, self.Fea_Layers, self.Fea_Filter, self.Fea_Block, self.Fea_Step) # 6 8 4 3
        self.feature  = AutoFeature(5, self.Fea_Layers, self.Fea_Filter, self.Fea_Block, self.Fea_Step, p=p)
        self.matching = AutoMatching(self.Mat_Layers, self.Mat_Filter, self.Mat_Block, self.Mat_Step, p=p)
        self.disp = Disp(self.maxdisp)
        # self.convltc = ConvLTC_inside_loop(input_c=5, output_c=32, kernel_size=3, stride=1, padding=1)
        self.snn_frontend = SCNN_frontend(input_c=5, output_c=16)


        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_p(self):
        self.feature.p = self.p
        self.feature.update_p()
        self.matching.p = self.p
        self.matching.update_p()

        
    def forward(self, x, y): 
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
                    l_r = 0
                else:
                    l_r = 2
                x_out = self.feature(x[:,i], left_or_right=l_r)
                y_out = self.feature(y[:,i], left_or_right=l_r+1)
                # print('y_out',y_out.shape)
                with torch.cuda.device_of(x_out):
                    matching_signature = self.concatinate(x_out, y_out) # 4, 24, 22, 88, 116
                    # print('matching_signature',matching_signature.shape)
                    # matching_signature = x_out.new().resize_(x_out.size()[0], x_out.size()[1]*2, self.maxdisp,  x_out.size()[2],  x_out.size()[3]).zero_()
                    # for  j in range(self.maxdisp):
                    #     if j > 0 :
                    #         matching_signature[:,:x_out.size()[1], j,:,j:] = x_out[:,:,:,j:]
                    #         matching_signature[:,x_out.size()[1]:, j,:,j:] = y_out[:,:,:,:-j]
                    #     else:
                    #         matching_signature[:,:x_out.size()[1],j,:,j:] = x_out
                    #         matching_signature[:,x_out.size()[1]:,j,:,j:] = y_out
                cost = self.matching(matching_signature, left_or_right=l_r).squeeze(1)
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
