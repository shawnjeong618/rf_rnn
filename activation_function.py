import torch
import torch.nn as nn
import numpy as np

import sys

# epsilon
eps = 1e-8


class RFRNNActivation(nn.Module):
    # initialization
    def __init__(self, hidden_size: int, v_b: float=-0.3, w_to_l: float=2.0, abs_v_th: float=0.3, v_dd: float=1.8):
        # initialize trainable module
        super().__init__()

        # set fixed hyper-parameters
        self.abs_v_th = abs_v_th    # PMOS threshold votlage (abs)
        self.v_dd = v_dd            # V_dd

        # # (case 1): trainable parameter
        # self.v_b = nn.Parameter((self.v_dd + self.abs_v_th) * torch.rand((hidden_size,))
        #                         - self.abs_v_th)
        # self.w_to_l = nn.Parameter(10 * torch.rand((hidden_size,)))

        # (case 2): fixed parameter
        self.register_buffer("v_b", torch.full(size=(hidden_size,), fill_value=v_b))
        self.register_buffer("w_to_l", torch.full(size=(hidden_size,), fill_value=w_to_l))

    # forward pass
    def forward(self, v_in):
        # # clamp bias voltage (critical for calculation)
        # v_b_clamped = self.v_b.clamp(min=-self.abs_v_th, max=self.v_dd)

        # activation region mask
        # mask_cond = torch.tensor(v_in >= (self.v_b + self.abs_v_th), dtype=torch.bool)

        # # (DEBUG 03122026)
        # print(v_b_clamped.shape)
        # print(mask_cond.shape)
        
        # # (DEBUG 03132026)
        # print(f"v_in device: {v_in.device}")
        # print(f"v_b device: {self.v_b.device}")
        
        # compute the first term
        base1 = (v_in - (self.v_b + self.abs_v_th))
        term1 = torch.pow(torch.abs(base1), 0.8)

        # compute the second term
        sqrt_arg = torch.clamp(2.0 * v_in, min=eps)
        base2 = (15 * torch.pi * self.w_to_l * torch.sqrt(sqrt_arg) / 16) - (self.v_b + self.abs_v_th)
        term2 = torch.sign(base2) * torch.pow(torch.abs(base2), 0.2)

        # use torch.where for piece-wise behavior
        y = torch.where(v_in > (self.v_b + self.abs_v_th), term1*term2, 0.0)

        # use torch.where to enforce operting voltage constraint
        y_clamped = torch.where(y > self.v_dd, self.v_dd, y)

        # (DEBUG 03132026)
        # print(f"v_in = {v_in}")
        # print(f"v_out = {y}")
        # print(f"v_out(clmaped) = {y_clamped}")
        

        return y_clamped

        # DEBUG
        # active_ratio = mask_cond.float().mean().item()
        # print("active ratio: ", active_ratio)
        # print("v_in min/max/mean:", v_in.min().item(), v_in.max().item(), v_in.mean().item())
        # print("v_in:", v_in.shape, v_in.dtype, v_in.device)
        # print("mask_cond:", mask_cond.shape, mask_cond.dtype, mask_cond.device)
        # print("w_to_l_m:", w_to_l_m.shape if torch.is_tensor(w_to_l_m) else type(w_to_l_m))
        # print("y :", y.shape)

class RFRNNActivation2(nn.Module):
    # initialization
    def __init__(self, hidden_size: int, v_b: float=-0.3, w_to_l: float=2.0, abs_v_th: float=0.3, v_dd: float=1.8):
        # initialize trainable module
        super().__init__()

        # (case 1) initialize trainable parameter
        self.v_b = nn.Parameter(torch.full((hidden_size,), v_b))
        # self.v_m = nn.Parameter(torch.full((hidden_size,), v_m))

        # (case 2) fixed parameters
        # self.register_buffer("v_b", torch.full((hidden_size, ), fill_value=v_b))
        self.register_buffer("w_to_l", torch.full((hidden_size, ), fill_value=w_to_l))

        # initialize hyper parameters
        self.abs_v_th = abs_v_th
        self.v_dd = v_dd

    # forward pass
    def forward(self, x):
        # compute active region
        active = self.w_to_l * (x - (self.v_b + self.abs_v_th))

        # (case 1) clamp lower bound
        y = torch.where(x >= self.v_b + self.abs_v_th, active, 0)

        # (case 2) clamp upper bound
        y = torch.where(y >= self.v_dd, self.v_dd, y)

        # (03182026) DEBUG
        #print(y)

        # piece-wise tensor
        return y
    

# TESTING
# hidden_size = 4
# init_v_b = -0.2
# act_fnc = RFRNNActivation(hidden_size=hidden_size, v_b=init_v_b)
# act_fnc_2 = RFRNNActivation2(hidden_size=hidden_size)

# batch_size = 2
# seq_len = 5

# data = 0.0 * torch.ones((batch_size, seq_len, hidden_size), requires_grad=True)

# x1 = torch.zeros_like(data[:, -1, :])
# x2 = torch.zeros_like(x1)
# y1 = torch.zeros_like(x1, requires_grad=True)
# y2 = torch.zeros_like(y1, requires_grad=True)
# for i in range(seq_len):
#     x_t = data[:, i, :]
#     y1 = act_fnc.forward(x_t + x1)
#     y2 = act_fnc_2.forward(x_t + x2)
#     x1 = x_t
#     x2 = x_t

# #print(data)
# print(y2)
# print(y2.backward(torch.ones_like(x1)), data.grad)