import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sys
sys.path.append('utils/MATLAB/')
sys.path.append('utils/preprocessing/')
sys.path.append('utils/recalibration/')
from recalibration_utils import *
from session_utils import *
from hmm_utils import *
from hmm import *
from preprocess import DataStruct, daysBetween


def get_DiscreteTargetGrid(struct, gridSize, task = None):
    '''Divide screen into a n x n grid of possible target locations. Inputs are: 

    struct (DataStruct) - session data to use 
    gridSize (int)      - number of rows and columns to chop the screen up into 
    task (str)          - task data to draw from; defaults to using all 

    TODO: update to work with new getTrainTest outputs, also deal with screen shifting around for different blocks
    '''

    if task is None:
        targpos_data = struct.targetPos_continuous
    else:
        targpos_data = np.concatenate([struct.targetPos[i] for i in np.where(struct.trialType == task)[0]])

    X_min, X_max  = targpos_data[:, 0].min() - 20, targpos_data[:, 0].max() + 20
    Y_min, Y_max  = targpos_data[:, 1].min() - 20, targpos_data[:, 1].max() + 20

    X_loc,Y_loc   = np.meshgrid(np.linspace(X_min, X_max, gridSize), np.linspace(Y_min, Y_max, gridSize))
    targLocs      = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T

    return targLocs 


def generateTargetGrid(gridSize, x_bounds = [-0.5, 0.5], y_bounds = [-0.5, 0.5]):
    '''
    Generate target grid for simulator.
    '''
    
    X_loc,Y_loc = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], gridSize), np.linspace(y_bounds[0], y_bounds[1], gridSize))
    targLocs    = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T
    
    return targLocs


def generateTransitionMatrix(gridSize, stayProb):
    '''
    Generate transition probability matrix for simulator targets.
    '''
    nStates     = gridSize**2
    stateTrans  = np.eye(nStates)*stayProb # Define the state transition matrix

    for x in range(nStates):
        idx                = np.setdiff1d(np.arange(nStates), x)
        stateTrans[x, idx] = (1-stayProb)/(nStates-1)

    pStateStart = np.zeros((nStates,1)) + 1/nStates
    
    return stateTrans, pStateStart


def prep_HMMData(struct, train_frac = 1., sigma = None, task = None, blocks = None, cutStart = None, return_flattened = False):
    '''
    Code for generating input data for HMM using session data. Inputs are:

    struct (DataStruct)      - session to train on
    train_frac (float)       - fraction of dataset to use on training 
    sigma (float)            - gaussian smoothing width (default: no smoothing)
    task (str)               - task type to train and test on
    blocks (str)             - blocks to use 
    cutStart (int)           - optionally remove # timepoints from beginning of each trial

    Returns:

    train/test_neural    - 
    train/test_cursorPos - 
    train/test_cursorErr - 

    TODO: 
    - join adjacent trials so that the returned list contains contiguous segments
    - figure out if returning individual trial lists to train_HMMRecalibrate causes bad performance (linear reg
      models show bad performance because the time snippets are so short)
    - maybe trash and just add optional return_cursorPos parameter to getTrainTest()
    '''
        
    neural, cursorErr, targPos       = getNeuralAndCursor(struct, sigma = sigma, task = task, blocks = blocks)
    
    n_trls                           = len(neural)
    train_ind, test_ind              = train_test_split(np.arange(n_trls), train_size = train_frac, shuffle = False)
    
    if cutStart is not None:
        neural    = [neural[i][cutStart:, :] for i in range(n_trls)]
        cursorErr = [cursorErr[i][cutStart:, :] for i in range(n_trls)]

    train_neural, test_neural        = [neural[i] for i in train_ind], [neural[i] for i in test_ind]
    train_cursorErr, test_cursorErr  = [cursorErr[i] for i in train_ind], [cursorErr[i] for i in test_ind]
    train_cursorPos, test_cursorPos  = [-1 * (cursorErr[i] - targPos[i]) for i in train_ind], [-1 * (cursorErr[i] - targPos[i]) for i in test_ind]
    
    if return_flattened:
        train_neural    = np.vstack(train_neural)
        test_neural     = np.vstack(test_neural)
        train_cursorPos = np.vstack(train_cursorPos)
        test_cursorPos  = np.vstack(test_cursorPos)
        train_cursorErr = np.vstack(train_cursorErr)
        test_cursorErr  = np.vstack(test_cursorErr)

    
    return train_neural, train_cursorPos, train_cursorErr, test_neural, test_cursorPos, test_cursorErr

