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



import sweep_utils


# %%%%%%%%%%%% Configurable parameters %%%%%%%%%%%%%%%%%%


sweepOpts                 = dict()
sweepOpts['model']        = ['PCA', 'FactorAnalysis']
sweepOpts['n_components'] = [2, 4, 6, 8, 10, 12]
sweepOpts['B']            = [80, 100, 120, 140, 160, 180, 190]
sweepOpts['thresh']       = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

parser = argparse.ArgumentParser(description = 'Code for optimizing HMM across session pairs.')
parser.add_argument('--participant', type = str, help = 'Participant ID (e.g. T5)')
parser.add_argument('--n_jobs', type = int, help = 'Number of jobs running this script')
parser.add_argument('--jobID', type = int, help = 'job ID')
parser.add_argument('--saveDir', type = str, default = './', help = 'Folder for saving scores')
args  = parser.parse_args()



# load dataset, add files as a sweep parameter:
DATA_DIR    = '/oak/stanford/groups/shenoy/ghwilson/nonstationarities/' + args.participant + '/train/'
SAVE_PATH   = args.saveDir + 'scores_ID_' + str(args.jobID) + '.npy'
files       = glob.glob(DATA_DIR + '*')
sweepOpts['file'] = files


# generate unchanging arguments for stabilizer
baseOpts = dict()



if __name__ == '__main__':
    np.random.seed(42)

    # split hyperparams list into chunks and select chunk that corresponds to this job ID
    sweep_args = sweep_utils.generateArgs(sweepOpts, baseOpts)
    sweep_args = np.array_split(sweep_args, args.n_jobs)[args.jobID]  # hack to fix remaining jobs
    
    print('*****Stabilizer sweep*****')
    print('Number of jobs: ', len(sweep_args))
    print('Number of CPUs: ', joblib.cpu_count())
    print('Running...')

    # if we have multiple CPUs, take advantage of them:
    if joblib.cpu_count() == 1:
        scores = list()
        for arg in sweep_args:
            scores.append(sweep_utils.test_Stabilizer(arg))
    else:
        scores = Parallel(n_jobs=-1, verbose = 0)(delayed(sweep_utils.test_Stabilizer)(arg) for arg in sweep_args)
        
    print('Done.')
    np.save(SAVE_PATH, scores)
    
    
   









