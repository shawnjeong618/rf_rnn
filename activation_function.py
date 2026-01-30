import torch
import torch.nn as nn

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
    def forward(self, v_in, w_to_l: float = 2.0, abs_v_th: float = 0.6):
        # input tensor dimension: [batch_size, sequence_length, input_size]
        

        # define piecewise funciton
        func2 = 