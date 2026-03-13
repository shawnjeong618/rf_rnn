import torch
import torch.nn as nn
import numpy as np

# epsilon
eps = 1e-6


class RFRNNActivation(nn.Module):
    # initialization
    def __init__(self, hidden_size: int, v_b: float, w_to_l: float=2.0, abs_v_th: float=0.3, v_dd: float=1.8):
        # initialize trainable module
        super().__init__()

        # set fixed hyper-parameters
        self.abs_v_th = abs_v_th    # PMOS threshold votlage (abs)
        self.v_dd = v_dd            # V_dd

        # initialize trainable parameter
        self.v_b = nn.Parameter((self.v_dd + self.abs_v_th) * torch.rand((hidden_size,))
                                - self.abs_v_th)
        self.w_to_l = nn.Parameter(10 * torch.rand((hidden_size,)))

    # forward pass
    def forward(self, v_in):
        # clamp bias voltage (critical for calculation)
        v_b_clamped = self.v_b.clamp(min=-self.abs_v_th, max=self.v_dd)
        v_b_b = v_b_clamped.unsqueeze(0).expand_as(v_in)
        w_to_l_b = self.w_to_l.unsqueeze(0).expand_as(v_in)

        # allocate output
        y = torch.zeros_like(v_in)

        # activation region mask
        mask_cond = v_in > (v_b_clamped + self.abs_v_th)

        # (DEBUG 03122026)
        print(v_b_clamped.shape)
        print(mask_cond.shape)

        # compute only for valid range
        if mask_cond.any():
            # elements of true conditions
            v_in_m = v_in[mask_cond]
            v_b_m = v_b_b[mask_cond]
            w_to_l_m = w_to_l_b[mask_cond]

            # compute the first term
            base1 = (v_in_m - (v_b_m + self.abs_v_th))
            term1 = base1.pow(0.8)

            # compute the second term
            sqrt_arg = (2.0 * v_in_m)
            base2 = (15 * torch.pi * w_to_l_m * torch.sqrt(sqrt_arg) / 16) - (v_b_m + self.abs_v_th)
            term2 = torch.sign(base2) * torch.abs(base2).pow(0.2)

            yt = (term1 * term2).clamp(min=0.0, max=self.v_dd)
            y = torch.where(mask_cond, yt, y)
        
        # DEBUG
        # active_ratio = mask_cond.float().mean().item()
        # print("active ratio: ", active_ratio)
        # print("v_in min/max/mean:", v_in.min().item(), v_in.max().item(), v_in.mean().item())
        # print("v_in:", v_in.shape, v_in.dtype, v_in.device)
        # print("mask_cond:", mask_cond.shape, mask_cond.dtype, mask_cond.device)
        # print("w_to_l_m:", w_to_l_m.shape if torch.is_tensor(w_to_l_m) else type(w_to_l_m))
        # print("y :", y.shape)

        return y
        # # define piecewise funciton
        # func_term1 = (v_in - (v_b_clamped + self.abs_v_th)) ** (4/5)
        # #func_term2 = ((self.v_b + self.abs_v_th) - (15 * torch.pi * self.w_to_l * torch.sqrt(2*v_in) / 16))
        # func_term2 = ((15*torch.pi*self.w_to_l*torch.sqrt(2*v_in)/16) - (v_b_clamped + self.abs_v_th))
        # func_term2 = torch.sign(func_term2) * torch.abs(func_term2) ** (1/5)

        # # define output tensors
        # y_t = func_term1 * func_term2
        # y_t = y_t.clamp(min=0.0, max=self.v_dd)
        # y_t_zeros = torch.zeros_like(y_t)

        # # return piece-wise tensor
        # return torch.where(v_in > self.v_b + self.abs_v_th, y_t, y_t_zeros)

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
        v_b_lim = torch.clamp(min=-self.abs_v_th, max=self.v_dd)
        v_m_lim = torch.clamp(min=eps)

        # piece-wise tensor
        return torch.relu(x - self.v_b) * self.v_m

        # try relu
        # return torch.relu(x)
    

# TESTING
hidden_size = 4
init_v_b = -0.2
act_fnc = RFRNNActivation(hidden_size=hidden_size, v_b=init_v_b)

batch_size = 2
seq_len = 100

data = torch.ones((batch_size, seq_len, hidden_size))
output = act_fnc.forward(data[:, -1, :])

print(output)