import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import copy, glob, sys, joblib, argparse
from joblib import Parallel, delayed

import os
HOME = os.path.expanduser('~')

[sys.path.append(f) for f in glob.glob(HOME + '/projects/nonstationarities/utils/*')]
from preprocess import DataStruct
from plotting_utils import figSize
from lineplots import plotsd
from session_utils import *
from recalibration_utils import *
from click_utils import *

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.model_selection import ParameterGrid
from hmm_utils import HMMRecalibration





def generateArgs(sweepOpts, baseOpts):
    '''Generates list of arguments for parallelizing adaptation runs across parameters.
       Inputs are:
       
           sweepOpts (dict) - parameters to sweep
           baseOpts (dict)  - unchanging settings '''
    
    args_list = list()
    grid      = ParameterGrid(sweepOpts)
    
    for arg_dict in grid:
        args_list.append({**arg_dict, **baseOpts})
    return args_list

        
def test_HMM(arg):
    '''Test HMM using generated session pairs dataset. Input <arg> is 
       dictionary with key-value pairs:

        'file'         : (str)          - path to session pair data to load
        'probWeighted' : (float or str) - probability threshold or 'probWeighted'
        'pStateStart'  : (2D float)     - nStates x 1 of prior target probabilities
        'stateTrans'   : (2D float)     - state transition matrix 
        'kappa'
        'inflection'
        'exp'
        '''
    
    pair_data = np.load(arg['file'], allow_pickle = True).item()
    HMM       = HMMRecalibration(arg['stateTrans'], pair_data['B_targLocs'], baseOpts['pStateStart'], arg['kappa'], 
                                 adjustKappa = lambda dist : 1 / (1 + np.exp(-1 * (dist - arg['inflection']) *arg['exp'])))
    
    decoder           = copy.deepcopy(pair_data['A_decoder'])
    train_neural      = [pair_data['B_train_neural']]
    train_cursorPos   = [pair_data['B_train_cursor']]
    test_neural       = np.vstack(pair_data['B_test_neural'])
    test_targvec      = np.vstack(pair_data['B_test_targvec'])
    
    new_decoder = HMM.recalibrate(decoder, train_neural, train_cursorPos)
    score       = new_decoder.score(test_neural, test_targvec)
    
    return score


# %%%%%%%%%%%% Configurable parameters %%%%%%%%%%%%%%%%%%

gridSize     = 20     # already got specified when making the datasets (possible coordinate diffs across sessions)
stayProb     = 0.999
probWeighted = 'probWeighted'

sweepOpts = dict()
sweepOpts['kappa']      = [0.5, 1, 2, 4, 6, 8]
sweepOpts['inflection'] = [0.1, 10, 30, 50, 70, 100, 200, 400]  
sweepOpts['exp']        = [1e-4, 1e-3, 0.025, 0.05, 0.1, 0.5, 1, 2, 4]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

parser = argparse.ArgumentParser(description = 'Code for optimizing HMM across session pairs.')
parser.add_argument('--participant', type = str, help = 'Participant ID (e.g. T5)')
parser.add_argument('--n_jobs', type = int, help = 'Number of jobs running this script')
parser.add_argument('--jobID', type = int, help = 'job ID')
parser.add_argument('--saveDir', type = str, default = './', help = 'Folder for saving scores')
args  = parser.parse_args()



remaining_jobs = [ 20,  32,  33,  56,  57,  58,  86,  99, 123, 152, 153, 166, 190,
       204, 214, 215, 219, 232, 233, 256, 257, 263, 286, 293, 299, 323,
       352, 353, 365, 366, 389, 390, 391, 419, 432, 456, 485, 486, 499,
       523, 552, 565, 566, 589, 590, 599]

# load dataset, add files as a sweep parameter:
DATA_DIR    = '/oak/stanford/groups/shenoy/gwilson/nonstationarities/' + args.participant + '/train/'
SAVE_PATH   = args.saveDir + 'scores_ID_' + str(remaining_jobs[args.jobID]) + '.npy'
files       = glob.glob(DATA_DIR + '*')
sweepOpts['file'] = files


# generate unchanging arguments for HMM 
nStates = gridSize**2
baseOpts                 = dict()
baseOpts['probWeighted'] = probWeighted
baseOpts['pStateStart']  = np.zeros((nStates,1)) + 1/nStates
baseOpts['stateTrans']   = np.eye(nStates)*stayProb 

for x in range(nStates):
    idx                            = np.setdiff1d(np.arange(nStates), x)
    baseOpts['stateTrans'][x, idx] = (1-stayProb)/(nStates-1)
    
    



if __name__ == '__main__':
    np.random.seed(42)
    
    # split hyperparams list into chunks and select chunk that corresponds to this job ID
    hmm_args = generateArgs(sweepOpts, baseOpts)
   # hmm_args = np.array_split(hmm_args, args.n_jobs)[args.jobID]
    hmm_args = np.array_split(hmm_args, 599)[remaining_jobs[args.jobID]]  # hack to fix remaining jobs

    
    print('Number of jobs: ', len(hmm_args))
    print('Number of CPUs: ', joblib.cpu_count())
    print('Running...')
    
 
    # if we have multiple CPUs, take advantage of them:
    if joblib.cpu_count() == 1:
        scores = list()
        for arg in hmm_args:
            scores.append(test_HMM(arg))
    else:
        scores = Parallel(n_jobs=-1, verbose = 0)(delayed(test_HMM)(arg) for arg in hmm_args)
    
    # generate a pandas dataframe for easy tracking of parameters and scores
    for hmm_arg, score in zip(hmm_args, scores):
        hmm_arg['score'] = score
    scores_df = pd.DataFrame(hmm_args)
    
    np.save(SAVE_PATH, scores_df)






















