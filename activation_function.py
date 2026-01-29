import torch
import torch.nn as nn

class RFRNNActivation(nn.Module):
    # initialization
    def __init__(self, hidden_size: int, init_v_b: float = 0.0):
        # initialize trainable module
        super().__init__()

        # set learnable parameters
        # (1) voltage bias
        self.v_b = nn.Parameter(torch.full((hidden_size,), init_v_b))

    # forward pass
    def forward(self, v_in, w_to_l: float = 2.0, abs_v_th: float = 0.6):
        # tensor dimension verification
        
        # define piecewise funciton
        func2 = 