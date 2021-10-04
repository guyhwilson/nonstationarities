import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, latent_size, n_layers = 1, dropout = 0):
        super(LSTM, self).__init__()
        self.input_size     = input_size 
        self.latent_size    = latent_size 
        self.dropout        = dropout 
        self.n_layers       = n_layers
      
        self.linear = nn.Linear(input_size, input_size)
        self.rnn    = nn.LSTM(input_size, latent_size, num_layers = n_layers, dropout = dropout, batch_first = False) 

    def forward(self, x, lengths):
        # input <x> is a tensor of size seqlen x batch x features
		x         = self.linear(x)
        x         = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
        x,_       = self.rnn(x)                                           
        x,_       = torch.nn.utils.rnn.pad_packed_sequence(x)
		
        return x
	
	
class Decoder(nn.Module):
    def __init__(self, latent_size, output_size, n_layers = 1, dropout = 0):
        super(LSTM, self).__init__()
        self.latent_size    = latent_size
        self.output_size    = output_size 
        self.dropout        = dropout 
        self.n_layers       = n_layers
      
        self.rnn    = nn.LSTM(latent_size, output_size, num_layers = n_layers, dropout = dropout, batch_first = False) 
		self.linear = nn.Linear(output_size, output_size)

    def forward(self, x, lengths):
        # input <x> is a tensor of size seqlen x batch x features
        x         = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
        x,_       = self.rnn(x)                                           
        x,_       = torch.nn.utils.rnn.pad_packed_sequence(x)
		x         = self.linear(x)
		
        return x
	
	
class Autoencoder():
	def __init__(self, observed_size, latent_size, n_layers = 1, dropout = 0):
        super().__init__()
        
        self.encoder  = Encoder(observed_size, latent_size, n_layers, dropout)
        self.decoder  = Decoder(latent_size, observed_size, n_layers, dropout)
        self.n_layers = n_layers
		self.dropout  = dropout
        
    def forward(self, x, lens):
        z    = self.encoder.forward(x, lens)
		xhat = self.decoder.forward(z, lens)
		
		return xhat 
		
       
