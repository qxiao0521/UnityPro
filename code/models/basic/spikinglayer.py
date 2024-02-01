import math
import torch
import torch.nn as nn

thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

class ActFun_b(torch.autograd.Function):
    # spike forward and backward function
    @staticmethod
    def forward(ctx, input, b):
        ctx.save_for_backward(input)
        ctx.b = b
        return input.gt(thresh).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        device = input.device
        grad_input = grad_output.clone()
        # temp = abs(input - thresh) < lens
        b = torch.tensor(ctx.b, device=device, dtype=input.dtype, requires_grad=False)
        temp = (1 - torch.tanh(b * (input-thresh)) ** 2) * b / 2 / torch.tanh(b/2)
        # temp = temp * (0<input<1)
        temp[input<=0]=0
        temp[input>=1]=0
        # return grad_output * temp.float(), None
        # temp = abs(input - thresh) < lens
        return grad_input * temp.to(grad_input.dtype), None, None

act_fun_b = ActFun_b.apply

class SpikeRelu(nn.Module):
    def __init__(self, mem_back=False):
        super(SpikeRelu, self).__init__()
        self.mem = None
        self.mem_back = mem_back

    def forward(self, x):
        if self.mem == None:
            self.mem = torch.zeros_like(x)
        mem = self.mem + x
        spike = act_fun(mem)
        self.mem = mem * decay * (1-spike)
        if self.mem_back:
            return spike, self.mem
        return spike

    def clear_mem(self):
        self.mem = None

class SpikeB(nn.Module):
    def __init__(self, b=3, mem_back=False):
        super().__init__()
        self.mem = None
        # self.spike_b = torch.tensor(b, requires_grad=False)
        self.spike_b = b
        # self.decay = nn.Parameter(torch.nn.init.constant_(torch.Tensor(1), 0.5))
        # self.act = ActFun_b.apply
        self.mem_back = mem_back

    def forward(self, x):
        if self.mem == None:
            self.mem = torch.zeros_like(x, dtype=x.dtype).requires_grad_(False)
            # print(self.mem.shape)
        # decay =  torch.clamp(self.decay, 0, 1)
        mem = self.mem + x
        spike = act_fun_b(mem, self.spike_b)
        # if self.training == False:
        #     print(x.dtype, self.mem.dtype, mem.dtype, spike.dtype)
        # self.mem = mem  * (1-spike) * torch.clamp(self.decay, 0, 1)
        self.mem = mem  * (1-spike) * decay
        if self.mem_back:
            return spike, self.mem
        return spike

    def clear_mem(self):
        self.mem = None

class MemB(nn.Module):
    def __init__(self, b=3):
        super().__init__()
        self.mem = None
        # self.spike_b = torch.tensor(b, requires_grad=False)
        self.spike_b = b

    def forward(self, x):
        if self.mem == None:
            self.mem = torch.zeros_like(x, dtype=x.dtype).requires_grad_(False)
            # print(self.mem.shape)
        mem = self.mem + x
        spike = act_fun_b(mem, self.spike_b)
        self.mem = (mem * decay * (1-spike))
        return self.mem

    def clear_mem(self):
        self.mem = None

class LTC(nn.Module):
    def __init__(self, b=3, in_channels=1):
        super().__init__()
        self.mem = None
        # self.spike_b = torch.tensor(b, requires_grad=False)
        # self.decay = nn.Parameter(torch.nn.init.constant_(torch.Tensor(in_channels), 0.2))
        self.decay = nn.Parameter(torch.nn.init.constant_(torch.Tensor(1), 0.2))
        self.spike_b = b

    def forward(self, x):
        if self.mem == None:
            self.mem = torch.zeros_like(x, dtype=x.dtype).requires_grad_(False)
            # print(self.mem.shape)
        mem = self.mem + x
        spike = act_fun_b(mem, self.spike_b)
        # self.mem = ((mem * (1-spike)).permute(0, 2, 3, 1) * torch.clamp(self.decay, 0, 1.0)).permute(0, 3, 1, 2)
        self.mem = (mem * (1-spike) * torch.clamp(self.decay, 0, 1))
        return self.mem

    def clear_mem(self):
        self.mem = None

class BinaryNeuron(nn.Module):
    def __init__(self, b=3, mem_back=False):
        super(BinaryNeuron, self).__init__()
        self.spike_b = b
        self.mem_back = mem_back

    def forward(self, x):
        spike = act_fun_b(x, self.spike_b)
        return spike