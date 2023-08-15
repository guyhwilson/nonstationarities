import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sys
#sys.path.append('utils/MATLAB/')
#sys.path.append('utils/preprocessing/')
#sys.path.append('utils/recalibration/')
from utils.recalibration.recalibration_utils import *
from utils.preprocessing.session_utils import *
from utils.recalibration.hmm import *


def get_DiscreteTargetGrid(struct, gridSize, task = None):
    '''  
    Divide screen into a n x n grid of possible target locations. Inputs are: 

    struct (DataStruct) - session data to use 
    gridSize (int)      - number of rows and columns to chop the screen up into 
    task (str)          - task data to draw from; defaults to using all 

    TODO: update to work with new getTrainTest outputs, also deal with screen shifting around for different blocks
    '''
    print('DEPRECATED - switch to generateTargetGrid()')


    X_loc,Y_loc   = np.meshgrid(np.linspace(X_min, X_max, gridSize), np.linspace(Y_min, Y_max, gridSize))
    targLocs      = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T

    return targLocs 


def generateTargetGrid(gridSize, is_simulated = False, struct = None, task = None):
    '''
    Generate target grid for simulator.
    '''
          
    assert np.logical_xor(is_simulated, struct != None), "<is_simulated> and <struct> parameters cannot both be toggled"
    
    if is_simulated:
        # if using simulator we know the screen's coordinates are enclosed in 1 x 1 box
        x_bounds = [-0.5, 0.5]
        y_bounds = [-0.5, 0.5]
        
    else:
        # if using real data, infer screen bounds from all past target positions 
        # as we don't have groundtruth from those days (unless we go into codebase and dig up)
        if task is None:
            targpos_data = struct.targetPos_continuous
        else:
            targpos_data = np.concatenate([struct.targetPos[i] for i in np.where(struct.trialType == task)[0]])

        x_bounds = [targpos_data[:, 0].min() - 20, targpos_data[:, 0].max() + 20]
        y_bounds = [targpos_data[:, 1].min() - 20, targpos_data[:, 1].max() + 20]
    
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


