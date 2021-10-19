import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from tcn import TemporalConvNet


    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers = 1, dropout = 0):
        super(LSTM, self).__init__()
        self.hidden_size    = hidden_size
        self.input_size     = input_size 
        self.output_size    = output_size 
        self.dropout        = dropout 
        self.n_layers       = n_layers
      
        self.dropout = nn.Dropout(p = dropout)
        self.linear  = nn.Linear(input_size, hidden_size)
        self.rnn     = nn.LSTM(hidden_size, output_size, num_layers = n_layers, dropout = 0 if n_layers == 1 else dropout, batch_first = False, bidirectional = False) 

    def forward(self, x, lengths):
        # input <x> is a tensor of size seqlen x batch x features
        x         = self.dropout(x)
        x         = self.linear(x)
        x         = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
        x,_       = self.rnn(x)                                           
        x,_       = torch.nn.utils.rnn.pad_packed_sequence(x)
        output    = torch.tanh(x)
        
        return output
    
    
class LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers = 1, dropout = 0):
        super(LSTM2, self).__init__()
        self.hidden_size    = hidden_size
        self.input_size     = input_size 
        self.output_size    = output_size 
        self.dropout        = dropout 
        self.n_layers       = n_layers
      
        self.dropout = nn.Dropout(p = dropout)
        self.rnn     = nn.LSTM(input_size, hidden_size, num_layers = n_layers, dropout = 0 if n_layers == 1 else dropout, batch_first = False, bidirectional = False) 
        self.linear  = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # input <x> is a tensor of size seqlen x batch x features
        x         = self.dropout(x)
        x         = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
        x,_       = self.rnn(x)                                           
        x,_       = torch.nn.utils.rnn.pad_packed_sequence(x)
        x         = self.linear(x)
        output    = torch.tanh(x)
        
        return output
    
    
    

class TCN(nn.Module):
    def __init__(self, input_size, classes, num_channels, kernel_size, dilation, dropout):
        super(TCN, self).__init__()
        self.tcn    = TemporalConvNet(input_size, num_channels, kernel_size, dilation = dilation, dropout=dropout)
        self.out    = nn.Linear(num_channels[-1], classes)

    def forward(self, x):
        # needs to be batch x channels x length to get into CNN
        #print(x.shape)
    
        output = self.tcn(x.permute(1, 2, 0)).permute(0, 2, 1)
        output = self.out(output)
  
        return output