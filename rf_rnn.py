import torch
import torch.nn as nn
from activation_function import RFRNNActivation     # custom activation function

class RFRNN(nn.Module):
    # class definition
    def __init__(self, hidden_size:int, seq_len:int, output_size:int, input_size:int = 1):
        # use RNN base class
        super(RFRNN, self).__init__()

        # model architecture parameter
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # activation function
        self.act_func = RFRNNActivation(hidden_size=hidden_size,
                                        init_v_b=0.0)

        # model parameter intialization
        # (1) W_t1: input-to-hidden_state weight
        self.wt1 = nn.Linear(in_features=self.input_size,
                             out_features=self.hidden_size,
                             bias=False)
        # (2) W_t2: hiddenstate-to-hidden_state weight
        self.wt2 = nn.Linear(in_features=self.hidden_size,
                             out_features=self.hidden_size,
                             bias=False)
        # bias parameter (V_b) is stored inside the activation function
        # as a trainable parameter.

    # forward function
    # the function assumes an input dimension of [batch_size, seq_len, hidden_size]
    def forward(self, x, hx=None):
        # analyze the input dimension
        batch_size, seq_len, _ = x.size()

        # assume hidden_state=0 unless it was set
        if hx is None:
            # zero-initialize the tensor (hidden_size,)
            hx = torch.zeros(batch_size, self.hidden_size)

        # copy hidden state
        h_t_minus_1 = hx.clone()

        # iteratevly forward the RNN sequence
        for t in range(self.seq_len):
            # get the first input
            input_t = x[:, t, :]

            # compute the output from the activation function and update the hidden state
            v_in = (self.wt1 @ input_t) + (self.wt2 @ h_t_minus_1)
            h_t_minus_1 = self.act_func.forward(v_in)
        
        # return the newest hidden state as the output
        return h_t_minus_1
    
# TEST

# variables
hidden_size = 4
output_size = 2
seq_len = 100
batch_size = 32

init_v_b = 1.6

# model initialization
rnn = RFRNN(hidden_size=hidden_size, seq_len=seq_len, output_size=output_size)