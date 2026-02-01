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
        # input tensor dimension: [batch_size, sequence_length, hidden_size] (batch_first=True)
        x_t = v_in

        # define piecewise funciton
        func_term1 = (x_t - (self.v_b + self.abs_v_th)) ** (4/5)
        func_term2 = ((self.v_b + self.abs_v_th) - (15 * np.pi * self.w_to_l * np.sqrt(2*x_t) / 16)) ** (1/5)
        y_t = func_term1 * func_term2

        # return piece-wise tensor
        return torch.where(x_t > self.v_b + self.abs_v_th, y_t, 0)
    

# TESTING
# hidden_size = 4
# init_v_b = 1.6
# act_fnc = RFRNNActivation(hidden_size=hidden_size, init_v_b=init_v_b)

# batch_size = 4
# seq_len = 100

# data = -torch.ones((batch_size, seq_len, hidden_size))
# output = act_fnc.forward(data[:, -1, :])
# print(output)