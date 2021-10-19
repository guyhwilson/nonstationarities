import argparse

import sys
import numpy as np
from scipy.io import loadmat, savemat
from copy import deepcopy

sys.path.append('../utils/preprocessing/')
from sklearn.linear_model import LinearRegression
import script_utils

parser = argparse.ArgumentParser(description = 'Supervised VKF recalibration. Returns weights matrix for linear decoder.')
parser.add_argument('--file', default = 'E:\Session\Data\HMMrecal\session_compressed.mat', type = str, help = 'Path to processed .mat file for training block(s)')
parser.add_argument('--saveDir', type = str, default = 'E:\Session\Data\HMMrecal/' , help = 'Folder for saving updated decoder weights.')
args  = parser.parse_args()



if __name__ == '__main__':

    data    = script_utils.loadCompressedSession(args.file)
    
    print('Recalibrating decoder...')
    PosErr  = data['targetPos'] - data['cursorPos']                     
    VKF     = LinearRegression(normalize = False).fit(data['TX'], PosErr) # TODO - align this with standard CL recal settings:
    weights = VKF.coef_ 
    savemat(args.saveDir + 'weights', {'weights' : weights})
    print('Done. Weights saved at: ', args.saveDir + 'weights.mat')
    
    