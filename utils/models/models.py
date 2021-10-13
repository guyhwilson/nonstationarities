import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from tcn import TemporalConvNet



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout = 0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size  = input_size 
        self.output_size = output_size 
        self.dropout     = dropout 
        self.n_layers    = 1

        self.rnn    = nn.LSTM(input_size, hidden_size, num_layers = 1, dropout = dropout, batch_first = False) 
        self.out    = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # if trying to integrate minibatch, add lengths arg to forward()
        # input <x> is a tensor of size seqlen x batch x features
        
        #a = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        #b = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        
        x,_       = self.rnn(x)                
        output    = self.out(x)
        return output
        
      
    #def init_hidden(self, batch_size, DEVICE):
    #   
    #  return (a, b)
  

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