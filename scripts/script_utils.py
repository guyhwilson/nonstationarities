import sys
import numpy as np
from scipy.io import loadmat, savemat
from copy import deepcopy

sys.path.append('../utils/preprocessing/')
sys.path.append('../utils/recalibration/')
sys.path.append('../utils/plotting/')
from hmm_utils import HMM_estimate
from plotting_utils import figSize
from sklearn.linear_model import LinearRegression



def loadCompressedSession(file):
    '''Loads and gets recent block/session data ready for HMM. Inputs are:
    
        file (str) - path to compressed R_Struct (this is output of processBlocks.m)
        
    TODO: cut out time prior to first trial to avoid non-BCI control timebins'''
    
    dat       = loadmat(file, simplify_cells = True)['compressed']
    dat['TX'] = dat['TX'].astype('float64')
    
    # cast as a list for edge case where matlab returns integer (single block): 
    if isinstance(dat['blockList'], int):
        dat['blockList'] = list(dat['blockList'])
    
    for block in dat['blockList']:
        blockMean                                = dat['TX'][dat['blockNums'] == block, :].mean(axis = 0)
        dat['TX'][dat['blockNums'] == block, :] -=  blockMean
        
    return dat



def get_CompressedDiscreteTargetGrid(compressed_session, gridSize):
    '''Return discretized BCI cursor screen locations for compressed block data.
       Inputs are:
       
           compressed_session (dict) - block/session data to use
           gridSize (int)            - grid width/height; total states = gridSize**2
    '''
    
    targpos_data  = compressed_session['targetPos']
    X_min, X_max  = targpos_data[:, 0].min() - 20, targpos_data[:, 0].max() + 20
    Y_min, Y_max  = targpos_data[:, 1].min() - 20, targpos_data[:, 1].max() + 20

    X_loc,Y_loc   = np.meshgrid(np.linspace(X_min, X_max, gridSize), np.linspace(Y_min, Y_max, gridSize))
    targLocs      = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T
    
    return targLocs