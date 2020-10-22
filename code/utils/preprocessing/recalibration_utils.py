import numpy as np
from sklearn.linear_model import LinearRegression


import sys
sys.path.append('utils/MATLAB/')
from HMM_utils import *


def getTrainTest(struct, train_frac = 0.5, task = 'cursor'):
  '''
  Code for getting training and test data. Inputs are:

      struct (DataStruct)      - session to train on
      train_frac (float)       - fraction of dataset to use on training 
      task (str)               - task type to train and test on   
  '''
  
  features             = np.vstack([struct.TX[i] for i in np.where(struct.trialType == task)[0] ])
  targets              = np.vstack([struct.targetPos[i] - struct.cursorPos[i] for i in np.where(struct.trialType == task)[0] ])
  n_samples, n_chans   = features.shape[0], features.shape[1]

  train_samples       = int(train_frac * n_samples)
  train_x, test_x     = features[:train_samples, :], features[train_samples:, :]
  train_y, test_y     = targets[:train_samples, :], targets[train_samples:, :]
  
  return train_x, test_x, train_y, test_y

  


def traintest_DecoderSupervised(struct, decoder = None, adapt_means = True, train_frac = 0.5, task = 'cursor'):
    '''
    Code for training linear decoder on session data. Inputs are:
        
        struct (DataStruct)      - session to train on
        decoder (Sklearn object) - decoder to use; if None then defaults to linear regression 
        adapt_means (Bool)       - if providing a decoder, whether or not to mean center the session data
        train_frac (float)       - fraction of dataset to use on training 
        task (str)               - task type to train and test on   
    '''
    
    train_x, test_x, train_y, test_y = getTrainTest(struct, train_frac = train_frac, task = task)
    session_means                    = train_x.mean(axis = 0)
    
    if decoder is None:
        decoder    = LinearRegression(fit_intercept = False, normalize = False).fit(train_x - session_means, train_y)
        test_score = decoder.score(test_x - session_means, test_y)
    else:
        if adapt_means:
            test_score = decoder.score(test_x - session_means, test_y)
        else:
            test_score = decoder.score(test_x, test_y)
    
    return test_score, decoder
  
  
  
def traintest_DecoderHMMUnsupervised(struct, decoder, stateTrans, pStateStart, targLocs, vmKappa, probThreshold = 0, train_frac = 0.5, task = 'cursor'):
    '''
    Code for training linear decoder on session data using HMM-inferred target locations. Inputs are:
        
        struct (DataStruct)      - session to train on
        decoder (Sklearn object) - decoder to use 
        stateTrans (2D array)    - transition probability matrix for HMM
        pStatestart (1D array)   - probability of beginning in a given state
        targLocs (2D arrays)     - corresponding locations on grid for targets
        vmKappa (float)          - precision parameter for the von mises distribution.
        probThreshold (float)    - threshold for subselecting high certainty regions (only where best guess > probThreshold)
        train_frac (float)       - fraction of dataset to use on training 
        task (str)               - task type to train and test on   
    '''
    
    train_x, test_x, train_y, test_y = getTrainTest(struct, train_frac = train_frac, task = task)
    train_samples                    = train_x.shape[0]
    session_means                    = train_x.mean(axis = 0)
    
    cursorPos            = np.vstack([struct.cursorPos[i] for i in np.where(struct.trialType == task)[0] ])
    targetPos            = np.vstack([struct.targetPos[i] - struct.cursorPos[i] for i in np.where(struct.trialType == task)[0] ])
    errorPos             = targetPos - cursorPos
    posTraj              = np.vstack([struct.cursorPos[i] for i in np.where(struct.trialType == task)[0] ])[:train_samples, :]
    rawDecTraj           = decoder.predict(train_x - session_means)
    
    targStates, logP = hmmviterbi_vonmises(rawDecTraj, stateTrans, targLocs, posTraj, pStateStart, vmKappa)
    pTargState, pSeq = hmmdecode_vonmises(rawDecTraj, stateTrans, targLocs, posTraj, pStateStart, vmKappa)

    # find time periods of high certainty:
    maxProb     = np.max(pTargState, axis = 0)
    highProbIdx = np.where(maxProb > probThreshold)[0]
    
    # now retrain HMM decoder, store for a new day:
    inferredTargLoc = targLocs[targStates.astype('int').flatten() - 1,:]
    inferredPosErr  = inferredTargLoc - cursorPos[:train_samples, :]
    decoder.fit((train_x - session_means)[highProbIdx, :], inferredPosErr[highProbIdx, :])
    
    # performance stats:
    test_score = decoder.score(test_x - session_means, test_y)
    r2         = np.diag(np.corrcoef(posTraj[highProbIdx, :].T, inferredTargLoc[highProbIdx, :].T), 2)
    
    return test_score, decoder, r2