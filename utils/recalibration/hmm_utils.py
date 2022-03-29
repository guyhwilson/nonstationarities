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


