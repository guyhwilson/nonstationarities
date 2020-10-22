import numpy as np
import sklearn 
from scipy.linalg import orthogonal_procrustes

import sys
#sys.path.append('')
from stabilizer_utils import fit_ConditionAveragedModel



def get_DimReduceData(struct, samples = 0, task = 'cursor'):
    '''If samples is 0, then use full trial data.
    
    '''
    trlens             = np.asarray([struct.TX[i].shape[0] for i in range(len(struct.TX))])
    trls               = np.where(np.logical_and(struct.trialType == task, trlens >= samples ))[0]
    
    if samples == 0:
        data = np.dstack([struct.TX[i] for i in trls])
    else:
      data     = np.dstack([struct.TX[i][:samples, :] for i in trls])
    
    unique, conditions = np.unique(np.vstack([struct.targetPos[i][0, :] for i in trls]), return_inverse = True, axis = 0)
    
    return data, conditions



def identifyGoodChannels(lambda_1, lambda_2, B, thresh):
  '''
  Greedy algorithm from Degenhart 2020 for identifying good channels. Inputs are:
    
    lambda_1 (2D array) - channels x factors loadings matrix for first dataset
    lambda_2 (2D array) - channels x factors loadings matrix for second dataset
    B (int)             - number of good channels desired
  '''
  
  channel_set  = np.arange(lambda_1.shape[0])
  
  below_thresh = np.where(np.logical_or(np.linalg.norm(lambda_1, axis = 1) < thresh, np.linalg.norm(lambda_1, axis = 1) < thresh))[0]
  channel_set  = np.setdiff1d(channel_set, below_thresh)
  
  while len(channel_set) > B:
    lambda_1prime = lambda_1[channel_set, :]
    lambda_2prime = lambda_2[channel_set, :]
    
    O, _        = orthogonal_procrustes(lambda_1prime, lambda_2prime)
    delta       = lambda_1prime.dot(O) - lambda_2prime
    worst       = np.argmax(np.linalg.norm(delta, axis = 1))
    
    channel_set = np.setdiff1d(channel_set, channel_set[worst])
    
  return channel_set
    


class Stabilizer(object):
  """
  Stabilizer object from Degenhart et al. 2020. Initialized with data
  from a reference session. Can then be fit to a new session to obtain 
  a transformation into the reference latent space. 
  """
  def __init__(self, model_type, n_components, dayA, task):
    '''
    Inputs are:

      model_type (str)      - 'PCA' or 'FactorAnalysis'
      n_components (int)    - latent space dimensionality
      dayA (DataStruct)     - reference session dataset
      task (string)       - task type to use
    '''
    self.model_type   = model_type
    self.n_components = n_components
    self.ref_task     = task

    dayA_data, conditions = get_DimReduceData(dayA, 20, self.ref_task)
    model, ll             = fit_ConditionAveragedModel(self.model_type, {'n_components' : self.n_components}, dayA_data, conditions)

    self.ref_model  = model
    self.ref_coefs  = model.components_.T


  def fit(self, dayB, task, B, thresh):
    '''Fit to a new session's data. Inputs are:
    
      dayB (DataStruct) - new session's dataset
      B (int)           - number of good channels desired
      thresh (float)    - cutoff for discarding noisy channels
    '''

    dayB_data, conditions = get_DimReduceData(dayB, 20, task)
    self.new_model, _     = fit_ConditionAveragedModel(self.model_type, {'n_components' : self.n_components}, dayB_data, conditions)
    self.new_coefs        = self.new_model.components_.T

    self.good_chans       = identifyGoodChannels(self.ref_coefs, self.new_coefs, B, thresh)
    self.R, _             = orthogonal_procrustes(self.new_coefs[self.good_chans, :], self.ref_coefs[self.good_chans, :])


  def transform(self, data): 
    '''Transform neural data into latent space of reference day. Inputs are:

      data (2D array) - time x channels
    '''

    newdata_latent = self.new_model.transform(data).dot(self.R)
    return newdata_latent
    