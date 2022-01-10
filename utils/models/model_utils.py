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


def prepareBatch(data_lists, ignore_index = 0, DEVICE = 'cpu'):
    ''' Given lists of data from a subset of trials, merge into a batch for 
    use in minibatched training. Inputs are:
    
        data_lists (list of lists) - sublists contain sample x channel arrays; by 
                                     convention assume first such list is input features
        ignore_index (float)       - padding for targets batch; can be used
                                     for automatic masking in e.g. CE loss
    '''
    n_lists = len(data_lists)
    
    data_lists_processed = list()
    for dl in data_lists:
        batch_size  = len(dl)
        maxlen      = max([a.shape[0] for a in dl])
        n_channels  = dl[0].shape[1]
        
        pad_dl      = np.zeros((maxlen, batch_size, n_channels)) # time x batch x  channels 

        lengths = list()
        for i in range(len(dl)):
            lengths.append(dl[i].shape[0])
            pad_dl[:lengths[-1], i, :] = dl[i].astype('float64')

        # mini-batch needs to be in decreasing order for pack_padded
        lengths = np.array(lengths)
        idx     = np.argsort(lengths)[::-1]
        lengths = lengths[idx]
        pad_dl   = pad_dl[:, idx, :]

        # convert to torch arrays and move onto device:
        lengths = torch.from_numpy(lengths).to(DEVICE)
        dl_batch = torch.tensor(pad_dl, requires_grad = True) 
        dl_batch = dl_batch.to(DEVICE)
        
        data_lists_processed.append(dl_batch)

    return tuple(data_lists_processed), lengths


      
def trainRNN(model, optimizer, x_list, y_list, batch_size, w_list = None, shuffle = True, DEVICE = 'cpu'):
    '''
    Helper function for training RNN (1 epoch). Inputs are:

    model (torch model)              - just use the above LSTM
    optimizer (torch optimizer)      - optimization alg
    x_list (list of channels x time) - features array
    Y_list (list of targets x time)  - targets array 
    batch_size (int)                 - minibatch size
    shuffle (Boolean)                - shuffle trial ordering if True
    DEVICE (idk)                     - optional GPU acceleration
    '''

    n_channels = x_list[0].shape[0]
    n_targets  = y_list[0].shape[0]
    n_trls     = len(x_list)

    model.train()
    for batch in minibatch_iterator(n_trls, batch_size, shuffle = shuffle):
        optimizer.zero_grad()
        
        if w_list is None:
            (batch_x, batch_y), lens = prepareBatch([[x_list[i] for i in batch], [y_list[i] for i in batch]], DEVICE = DEVICE)
            batch_w                  = None
        else:
            batch_lists = [[x_list[i] for i in batch], [y_list[i] for i in batch], [w_list[i] for i in batch]]
            (batch_x, batch_y, batch_w), lens = prepareBatch(batch_lists, DEVICE = DEVICE)

        output  = model.forward(batch_x, lens) # output is: time x batch x classses
        loss    = evaluate_batch_loss(output, batch_y, lens, weights = batch_w, DEVICE = DEVICE)
        
        loss.backward()
        optimizer.step()

    if DEVICE != 'cpu':
        del loss, output, batch_x, batch_y
        torch.cuda.empty_cache()
        
        
def evaluate_batch_loss(preds, targs, lens, weights = None, DEVICE = 'cpu'):
    
    loss         = 0
    loss_mask    = torch.zeros(preds.shape, device = DEVICE)
    for i in range(preds.shape[1]):
        loss_mask[:lens[i], i, :]  = 1.

    loss = ((torch.flatten(targs) - torch.flatten(preds)) ** 2.0) 
    loss = loss * torch.flatten(loss_mask)
    
    if weights is not None:
        loss = loss * torch.flatten(weights)
        
    loss = torch.sum(loss) / torch.sum(loss_mask)
    
    return loss
               
    
    
  
def evaluateRNN(model, x_list, y_list, batch_size, DEVICE = 'cpu'):
  '''Helper function for evaluating RNN performance. Inputs are:
  
    model (torch model)              - just use the above LSTM  
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
  
    
  model.eval()
  for batch in minibatch_iterator(n_trls, batch_size, shuffle = False):
    (batch_x, batch_y), lens = prepareBatch([[x_list[i] for i in batch], [y_list[i] for i in batch]], DEVICE = DEVICE)
    output                   = model.forward(batch_x, lens)  # output is: time x batch x classses
    
    total_loss += evaluate_batch_loss(output, batch_y, lens, weights = None, DEVICE = DEVICE).cpu().item() * output.shape[1]
     
    if DEVICE != 'cpu':
      del output, batch_x, batch_y
      torch.cuda.empty_cache()
        
  mean_loss = total_loss / n_trls
    
  return mean_loss 




def predictRNN(model, x_list, DEVICE = 'cpu'):
  '''Helper function for evaluating RNN performance. Inputs are:
  
    model (torch model)              - just use the above LSTM
    X_array (channels x time x trls) - features array
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
      (batch_x), lens = prepareBatch([[x_list[trl]]], DEVICE = DEVICE)
      output          = model.forward(batch_x[0], lens)  
      pred            = output[:lens[0], :, :].cpu().numpy() 
      preds.append(pred)
    
      del output, batch_x, lens
      torch.cuda.empty_cache()
    
  return preds