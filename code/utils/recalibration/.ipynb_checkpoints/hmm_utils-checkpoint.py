import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sys
sys.path.append('utils/MATLAB/')
sys.path.append('utils/preprocessing/')
from HMM_matlab import *
from preprocess import DataStruct, daysBetween


def traintest_DecoderHMMUnsupervised(struct, decoder, stateTrans, pStateStart, targLocs, vmKappa, 
                                     probThreshold = 0., train_frac = 0.5, task = None):
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
        
    TODO: update to work with new getTrainTest outputs 
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
    
    # if data too noisy or sessions too far apart, HMM may not have any valid high confidence time points
    # so we decrement the threshold by 0.1 until valid samples exist 
    isFitted        = False
    while not isFitted:
      try:
        decoder.fit((train_x - session_means)[highProbIdx, :], inferredPosErr[highProbIdx, :])
        isFitted = True
      except:
        probThreshold -= 0.1
        highProbIdx    = np.where(maxProb > probThreshold)[0]
        print('ProbThreshold too high. Lowering by 0.1')
      
    
    # performance stats:
    test_score = decoder.score(test_x - session_means, test_y)
    r2         = np.diag(np.corrcoef(posTraj[highProbIdx, :].T, inferredTargLoc[highProbIdx, :].T), 2)
    
    #return posTraj[:, :].T, inferredTargLoc[:, :].T
    return test_score, decoder, r2