import torch
import torch.nn as nn
import numpy as np


class RFRNNActivation(nn.Module):
    # initialization
    def __init__(self, hidden_size: int, init_v_b: float, w_to_l: float=2.0, abs_v_th: float=0.6):
        # initialize trainable module
        super().__init__()

        # set fixed hyper-parameters
        self.w_to_l = w_to_l        # width-to-length ratio
        self.abs_v_th = abs_v_th    # PMOS threshold votlage (abs)

        # initialize trainable parameter
        self.v_b = nn.Parameter(torch.full((hidden_size,), init_v_b))

    # forward pass
    def forward(self, v_in):
        # activation condition
        cond = (v_in > self.v_b + self.abs_v_th)

        # define piecewise funciton
        func_term1 = (x_t - (self.v_b + self.abs_v_th)) ** (4/5)
        func_term2 = ((self.v_b + self.abs_v_th) - (15 * torch.pi * self.w_to_l * torch.sqrt(2*x_t) / 16)) ** (1/5)
        y_t = func_term1 * func_term2
        y_t_zeros = torch.zeros_like(y_t)

        # DEBUG
        # print(func_term1)
        # print(func_term2)

        # return piece-wise tensor
        return torch.where(x_t > self.v_b + self.abs_v_th, y_t, y_t_zeros)

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
# hidden_size = 4
# init_v_b = 0.5
# init_v_m = 2.0
# act_fnc = RFRNNActivation2(hidden_size=hidden_size, v_b=init_v_b, v_m=init_v_m)

# batch_size = 8
# seq_len = 100

# data = 0.4*torch.ones((batch_size, seq_len, hidden_size))
# output = act_fnc.forward(data[:, -1, :])

# print(output)