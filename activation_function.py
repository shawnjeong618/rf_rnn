import torch
import torch.nn as nn
import numpy as np

import sys

# epsilon
eps = 1e-6


class RFRNNActivation(nn.Module):
    # initialization
    def __init__(self, hidden_size: int, v_b: float=0.5, w_to_l: float=2.0, abs_v_th: float=0.3, v_dd: float=1.8):
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
        term1 = torch.abs(base1).pow(0.8)

        # compute the second term
        sqrt_arg = torch.clamp(2.0 * v_in, min=0.0)
        base2 = (15 * torch.pi * self.w_to_l * torch.sqrt(sqrt_arg) / 16) - (self.v_b + self.abs_v_th)
        term2 = torch.sign(base2) * torch.abs(base2).pow(0.2)

        # use torch.where for piece-wise behavior
        y = torch.where(v_in >= (self.v_b + self.abs_v_th), term1*term2, 0.0)

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
    def __init__(self, hidden_size: int, v_b: float=0.0, v_m: float=1.0, abs_v_th: float=0.3, v_dd: float=1.8):
        # initialize trainable module
        super().__init__()

        # initialize trainable parameter
        self.v_b = nn.Parameter(torch.full((hidden_size,), v_b))
        self.v_m = nn.Parameter(torch.full((hidden_size,), v_m))

        # initialize hyper parameters
        self.abs_v_th = abs_v_th
        self.v_dd = v_dd

    # forward pass
    def forward(self, x):
        # method(1): hard-clamp the parameters
        v_b_lim = self.v_b.clamp(min=-self.abs_v_th, max=self.v_dd)
        v_m_lim = self.v_m.clamp(min=eps)

        # piece-wise tensor
        return torch.relu(x - self.v_b) * self.v_m

        # try relu
        # return torch.relu(x)
    

# TESTING
# hidden_size = 4
# init_v_b = -0.2
# act_fnc = RFRNNActivation(hidden_size=hidden_size, v_b=init_v_b)
# act_fnc_2 = RFRNNActivation2(hidden_size=hidden_size)

# batch_size = 2
# seq_len = 100

# data = torch.rand((batch_size, seq_len, hidden_size), requires_grad=True)

# x_prev = torch.zeros_like(data[:, -1, :])
# y = torch.zeros_like(x_prev, requires_grad=True)
# for i in range(seq_len):
#     x_t = data[:, i, :]
#     y = act_fnc.forward(x_t + x_prev)
#     x_prev = x_t

# print(data)
# print(y)
# print(y.backward(torch.ones_like(data)), data.grad)