import argparse

import sys
import numpy as np
from scipy.io import loadmat, savemat
from copy import deepcopy

sys.path.append('../utils/preprocessing/')
sys.path.append('../utils/recalibration/')
sys.path.append('../utils/plotting/')
from hmm_utils import HMM_estimate
from hmm import HMMRecalibration
from plotting_utils import figSize
from sklearn.linear_model import LinearRegression
import script_utils


parser = argparse.ArgumentParser(description = 'Hmm-based VKF recalibration. Returns weights matrix for linear decoder.')
parser.add_argument('--file', default = 'E:\Session\Data\HMMrecal\session_compressed.mat', type = str, help = 'Path to processed .mat file for training block(s)')
parser.add_argument('--saveDir', type = str, default = 'E:\Session\Data\HMMrecal/' , help = 'Folder for saving updated decoder weights.')
parser.add_argument('--kappa', type = float, default = 2., help = 'Dispersion parameter for HMM observation model')
parser.add_argument('--gridSize', type = int, default = 20, help = 'Number of rows/columns to divide screen up into')
args  = parser.parse_args()



def adjustKappa(dist):
    coef = 1 / (1 + np.exp(-1 * (dist - 50) * 0.1))
    return coef 


if __name__ == '__main__':
    stayProb      = 0.999
    vmKappa       = args.kappa
    gridSize      = args.gridSize
    probThreshold = 'probWeighted'

    #---------------------------------------------

    data          = script_utils.loadCompressedSession(args.file)
    targLocs      = script_utils.get_CompressedDiscreteTargetGrid(data, gridSize = gridSize)
    nStates       = gridSize**2
    
    # Define state transition matrix and prior state probabilities 
    stateTrans    = np.eye(nStates)*stayProb 
    for x in range(nStates):
        idx                = np.setdiff1d(np.arange(nStates), x)
        stateTrans[x, idx] = (1-stayProb)/(nStates-1)

    pStateStart = np.zeros((nStates,1)) + (1/nStates)
    
    hmm = HMMRecalibration(stateTrans, targLocs, pStateStart, vmKappa, adjustKappa)

    # heavy lifting - infer hidden state sequences and probabilities
    print('Inferring target positions...')
    targStates = hmm.viterbi_search(data['decVel'], data['cursorPos'])[0]
    pTargState = hmm.decode(data['decVel'], data['cursorPos'])[0]
    print('Recalibrating decoder...')

    maxProb         = np.max(pTargState, axis = 0)              
    inferredTargLoc = targLocs[targStates.astype('int').flatten(), :]    # find predicted target locations
    inferredPosErr  = inferredTargLoc - data['cursorPos']                # generate inferred cursorErr signals

    # TODO - align this with standard CL recal settings:
    if probThreshold == 'probWeighted':
        VKF = LinearRegression(normalize = False).fit(data['TX'], inferredPosErr, maxProb**2)
    else:
        raise ValueError('Discrete cut-off not currently supported.')


    weights = VKF.coef_ 
    savemat(args.saveDir + 'weights', {'weights' : weights})
    print('Done. Weights saved at: ', args.saveDir + 'weights.mat')
    
    