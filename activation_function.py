import torch
import torch.nn as nn
import numpy as np


class RFRNNActivation(nn.Module):
    # initialization
    def __init__(self, hidden_size: int, v_b: float, w_to_l: float=2.0, abs_v_th: float=0.6):
        # initialize trainable module
        super().__init__()

        # set fixed hyper-parameters
        self.w_to_l = w_to_l        # width-to-length ratio
        self.abs_v_th = abs_v_th    # PMOS threshold votlage (abs)

        # initialize trainable parameter
        self.v_b = nn.Parameter(torch.full((hidden_size,), v_b))

    # forward pass
    def forward(self, v_in):
        # activation condition
        cond = (v_in > self.v_b + self.abs_v_th)

        # define piecewise funciton
        func_term1 = (v_in - (self.v_b + self.abs_v_th)) ** (4/5)
        func_term2 = ((self.v_b + self.abs_v_th) - (15 * torch.pi * self.w_to_l * torch.sqrt(2*v_in) / 16)) ** (1/5)
        y_t = func_term1 * func_term2
        y_t_zeros = torch.zeros_like(y_t)

        # DEBUG
        a = self.v_b + self.abs_v_th
        b = 15.0 * torch.pi * self.w_to_l * torch.sqrt(2.0*v_in) / 16.0
        print(a)
        print(b)
        print((a-b)**(1/5))

        # return piece-wise tensor
        return torch.where(v_in > self.v_b + self.abs_v_th, y_t, y_t_zeros)

class RFRNNActivation2(nn.Module):
    # initialization
    def __init__(self, hidden_size: int, v_b: float=0.0, v_m: float=1.0):
        # initialize trainable module
        super().__init__()

        # initialize trainable parameter
        self.v_b = nn.Parameter(torch.full((hidden_size,), v_b))
        self.v_m = nn.Parameter(torch.full((hidden_size,), v_m))

    # forward pass
    def forward(self, x):
        # piece-wise tensor
        return torch.relu(x - self.v_b) * self.v_m

        # try relu
        # return torch.relu(x)
    

# TESTING
hidden_size = 1
init_v_b = 69.0
act_fnc = RFRNNActivation(hidden_size=hidden_size, v_b=init_v_b)

batch_size = 1
seq_len = 100

data = 70.0 * torch.ones((batch_size, seq_len, hidden_size))
output = act_fnc.forward(data[:, -1, :])

print(output)