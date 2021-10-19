import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import torch
from torch.autograd import Variable


def minibatch_iterator(n_trls, batch_len, shuffle = True):
    '''
    Return a list containing randomized minibatch sets.
    '''
    trls    = np.arange(n_trls)
    if shuffle:
        trls    = trls[np.random.permutation(n_trls)]   # randomize for each epoch to avoid cyclic stuff that rarely happesn but whatever
    batches = list()
    
    for i in range(0, n_trls, batch_len):
        batches.append(trls[i:i+batch_len])   
    return batches
  
  
def prepareBatch(x_list, y_list, ignore_index = 0, DEVICE = 'cpu', addNoise = 0):
  '''
  Given lists of data from a subset of trials, merge into a batch for 
  use in minibatched training. Inputs are:
  
  x_list (list of samples x inputs arrays)   - neural data list
  y_list (list of samples x outputs vectors) - corresponding outputs
  ignore_index (float)                       - padding for targets batch; can be used
                                               for automatic masking in e.g. CE loss
  '''
  # todo erase this annoying convention diff
  x_list = [x.T for x in x_list]
  y_list = [y.T for y in y_list]
    
  # determine the maximum time length per batch and zero pad the tensors
    

  n_max     = max([a.shape[1] for a in x_list])
  pad_x     = np.zeros((x_list[0].shape[0], n_max, len(x_list)))   # channels x time x trls
  pad_y     = np.ones((y_list[0].shape[0], n_max, len(x_list))) * ignore_index 
  lengths   = []
    
  for i in range(len(x_list)):
    lengths.append(x_list[i].shape[1])
    pad_x[:, :int(x_list[i].shape[1]), i] = x_list[i].astype('float64')
    
    if y_list is not None:
      pad_y[:, :int(x_list[i].shape[1]), i] = y_list[i]
        
  # mini-batch needs to be in decreasing order for pack_padded
  lengths = np.array(lengths)
  idx     = np.argsort(lengths)[::-1]
  lengths = lengths[idx]
  pad_x   = pad_x[:, :, idx]
  
  if y_list is not None:
    pad_y = pad_y[:, :, idx]
  
  if addNoise > 0:
    pad_x += np.random.normal(0, addNoise, pad_x.shape)
                  
  # convert to torch arrays and move onto device:
  lengths = torch.from_numpy(lengths).to(DEVICE)
  x_batch = Variable(torch.from_numpy(np.transpose(pad_x, (1, 2, 0)) ))
  x_batch = x_batch.to(DEVICE)
  
  if y_list is not None:
    y_batch = torch.from_numpy(np.transpose(pad_y, (1, 2, 0)) )
    y_batch = y_batch.to(DEVICE)
  
    return x_batch, y_batch, lengths
  
  else:
    return x_batch, lengths

      
def trainRNN(model, loss_fcn, optimizer, x_list, y_list, batch_size, ignore_index, shuffle = True, DEVICE = 'cpu',  addNoise = 0):
  '''
  Helper function for training RNN (1 epoch). Inputs are:
  
  model (torch model)              - just use the above LSTM
  loss_fcn (torch loss fcn)        - define loss
  optimizer (torch optimizer)      - optimization alg
  x_list (list of channels x time) - features array
  Y_list (list of targets x time)  - targets array 
  batch_size (int)                 - minibatch size
  ignore_index (int)               - label used for masking out time frames in loss 
  shuffle (Boolean)                - shuffle trial ordering if True
  DEVICE (idk)                     - optional GPU acceleration
  '''
  
  n_channels = x_list[0].shape[0]
  n_trls     = len(x_list)
  n_targets  = y_list[0].shape[0]
    
  model.train()
  for batch in minibatch_iterator(n_trls, batch_size, shuffle = shuffle):
    optimizer.zero_grad()
    batch_x, batch_y, lens = prepareBatch([x_list[i] for i in batch], [y_list[i] for i in batch], ignore_index= ignore_index, DEVICE = DEVICE, addNoise = addNoise)
    output                 = model.forward(batch_x, lens) # output is: time x batch x classses
  
    loss         = 0
    for i in range(batch_x.shape[1]):
      trl_output = output[:lens[i], i, :]
      trl_actual = batch_y[:lens[i], i, :]
      loss      += loss_fcn(trl_output, trl_actual)  # need batch x classes x extra dims 
    
    loss.backward()
    optimizer.step()
    
    if DEVICE != 'cpu':
      del loss, output, batch_x, batch_y
      torch.cuda.empty_cache()
    
    
def evaluateRNN(model, loss_fcn, x_list, y_list, batch_size, ignore_index = -100, DEVICE = 'cpu'):
  '''Helper function for evaluating RNN performance. Inputs are:
  
    model (torch model)              - just use the above LSTM
    loss_fcn (torch loss fcn)        - define loss
  
    X_array (channels x time x trls) - features array
    Y_array (targets x time x trls)  - targets array 
    DEVICE (idk)                     - optional GPU acceleration
    
    Returns:
    
    trls_performance (list) - entries contain performance on each trial
  '''
  
  n_channels = x_list[0].shape[0]
  n_trls     = len(x_list)
  n_targets  = y_list[0].shape[0]
  
  total_loss = 0
  total_corr = 0
  
    
  model.eval()
  for batch in minibatch_iterator(n_trls, batch_size, shuffle = False):
    batch_x, batch_y, lens = prepareBatch([x_list[i] for i in batch], [y_list[i] for i in batch], ignore_index= ignore_index, DEVICE = DEVICE, addNoise = 0)
    output                 = model.forward(batch_x, lens)                # output is: time x batch x classses
    
    # super inefficient lol:
    for i in range(batch_x.shape[1]):
      trl_output  = output[:lens[i], i, :]
      trl_actual  = batch_y[:lens[i], i, :]
      loss        = loss_fcn(trl_output, trl_actual)  # need batch x classes x extra dims 
      total_loss += loss.cpu().item()
      
     
      corr        = np.corrcoef(trl_output.detach().cpu().numpy().flatten(), trl_actual.cpu().numpy().flatten())[1, 0]
      total_corr += corr
    
    if DEVICE != 'cpu':
      del loss, output, batch_x, batch_y, trl_output, trl_actual
      torch.cuda.empty_cache()
        
  mean_corr = total_corr / n_trls
  mean_loss = total_loss / n_trls
    
  return mean_loss, mean_corr




def predictRNN(model, x_list, y_list, ignore_index = -100, DEVICE = 'cpu'):
  '''Helper function for evaluating RNN performance. Inputs are:
  
    model (torch model)              - just use the above LSTM
    X_array (channels x time x trls) - features array
    Y_array (targets x time x trls)  - targets array 
    DEVICE (idk)                     - optional GPU acceleration
    
    Returns:
    preds (list) - contains trial predictions
  '''
  
  n_trls = len(x_list)
  preds     = []
  raw_probs = []
  
  model.eval()
  with torch.no_grad():
    for trl in range(n_trls):
      batch_x, _, lens  = prepareBatch([x_list[trl]] , [y_list[trl]], ignore_index= ignore_index, DEVICE = DEVICE, addNoise = 0)
      output            = model.forward(batch_x, lens)  
      pred              = output[:lens[0], :, :].cpu().numpy() 
      preds.append(pred)
    
      del output, batch_x, lens
      torch.cuda.empty_cache()
    
  return preds