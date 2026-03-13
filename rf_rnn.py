import torch
import torch.nn as nn
from activation_function import RFRNNActivation     # custom activation function
from activation_function import RFRNNActivation2    # simplified custom activation function

class RFRNN(nn.Module):
    # class definition
    def __init__(self, hidden_size:int, seq_len:int, output_size:int, input_size:int = 1):
        # use RNN base class
        super(RFRNN, self).__init__()

        # model architecture parameter
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.output_size = output_size

        # activation function
        self.act_func = RFRNNActivation(hidden_size=hidden_size, v_b=0.2, w_to_l=2.0)
        #self.act_func = RFRNNActivation2(hidden_size=hidden_size, v_b=0.0, v_m=2.0)

        # model parameter intialization
        # (1) W_t1: input-to-hidden_state weight
        self.wt1 = nn.Linear(in_features=self.input_size,
                            out_features=self.hidden_size,
                            bias=False)
        # (2) W_t2: hidden_state-to-hidden_state weight
        self.wt2 = nn.Linear(in_features=self.hidden_size,
                            out_features=self.hidden_size,
                            bias=False)
        # bias parameter (V_b) is stored inside the activation function
        # as a trainable parameter.
        # (3) h_o: hidden_state-to-output weight
        self.h_o = nn.Linear(in_features=self.hidden_size,
                            out_features=self.output_size,
                            bias=False) 
        
        # (03052026)
        # (4) b: bias term
        # self.b = nn.Parameter()

    # forward function
    # the function assumes an input dimension of [batch_size, seq_len, hidden_size]
    def forward(self, x, h_t_minus_1=None):
        # analyze the input dimension
        batch_size, _, _ = x.shape

        # assume hidden_state=0 unless it was set
        if h_t_minus_1 is None:
            # zero-initialize the tensor (hidden_size,)
            h_t_minus_1 = torch.zeros(batch_size,
                                    self.hidden_size,
                                    device=x.device,
                                    dtype=x.dtype)

        # iteratevly forward the RNN sequence
        for t in range(self.seq_len):
            # get the first input
            input_t = x[:, t, :]

            # compute the output from the activation function and update the hidden state
            v_in = self.wt1(input_t) + self.wt2(h_t_minus_1)
            h_t_minus_1 = self.act_func.forward(v_in)

            # (DEBUG: 03052026) V_in, V_out tracking
            #print(f"v_in[{t}] = ", v_in)
            #print(f"v_out[{t}] = ", h_t_minus_1)
        
        # perform linear tranformation of hidden_state to output
        return self.h_o(h_t_minus_1)

    # weight initialization
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# # variables
# input_size = 1
# hidden_size = 4
# output_size = 2
# seq_len = 100
# batch_size = 32
# init_v_b = 0.2

# # model initialization
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"
# model = RFRNN(hidden_size=hidden_size, seq_len=seq_len, output_size=output_size).to(device)
# input = -2*torch.ones(batch_size, seq_len, input_size).to(device)     # input
# model.eval()        # evaluation mode
# output = model(input)
# print(output)