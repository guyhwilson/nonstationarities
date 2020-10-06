import numpy as np
import sklearn 
from scipy.linalg import orthogonal_procrustes



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
    #print(lambda_1.shape)
    #print(channel_set)
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
    def __init__(self, dimreduce, ):
      '''
      Inputs are:
      
        dimreduce (dimensionality reduction object) - any dimensionality reduction technique, must have  
                                                        fit() and transform() methods in sklearn format
      '''
      
      
      
      def fit():
        '''
        Inputs are:
        
          s (list of ints) - stable channel set
        '''
        return
        
        
    def transform(): 
      return