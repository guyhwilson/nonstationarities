import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import copy, glob, sys, joblib, argparse
from joblib import Parallel, delayed
import re

import os
HOME = os.path.expanduser('~')

[sys.path.append(f) for f in glob.glob(HOME + '/projects/nonstationarities/utils/*')]
from preprocess import DataStruct
from plotting_utils import figSize
from lineplots import plotsd
from session_utils import *
from recalibration_utils import *

import tensorflow.compat.v1 as tf
import sweep_utils


parser = argparse.ArgumentParser(description = 'Code for optimizing HMM across session pairs.')
parser.add_argument('--participant', type = str, help = 'Participant ID (e.g. T5)')
parser.add_argument('--n_jobs', type = int, help = 'Number of jobs running this script')
parser.add_argument('--jobID', type = int, help = 'job ID')
parser.add_argument('--saveDir', type = str, default = './', help = 'Folder for saving scores')
args  = parser.parse_args()


# %%%%%%%%%%%% Configurable parameters %%%%%%%%%%%%%%%%%%

sweepOpts = dict()
sweepOpts['n_epochs']   = [200]
sweepOpts['batch_size'] = [16, 32, 64]
sweepOpts['d_lr']       = [5e-6, 1e-5, 5e-5] 
sweepOpts['g_lr']       = [1e-6, 5e-6, 1e-5]
DATA_DIR   = '/oak/stanford/groups/shenoy/ghwilson/nonstationarities/' + args.participant + '/train/'


#sweepOpts['batch_size'] = [16]
#sweepOpts['d_lr']       = [5e-6] 
#sweepOpts['g_lr']       = [5e-6]
#DATA_DIR = f'/oak/stanford/groups/shenoy/ghwilson/nonstationarities/{args.participant}/session_pairs/'

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
np.random.seed(42)
tf.set_random_seed(seed=42)

# load dataset, add files as a sweep parameter:
model_folder = '/oak/stanford/groups/shenoy/ghwilson/nonstationarities/T5/ADAN/models/'
SAVE_PATH    = args.saveDir + 'scores_ID_' + str(args.jobID) + '.npy'
files        = glob.glob(DATA_DIR + '*')
#sweepOpts['file'] = files

sweepOpts['file'] = np.random.choice(files, 30, replace = False)



# generate unchanging arguments for stabilizer
baseOpts = dict()
baseOpts['spike_dim']  = 192
baseOpts['latent_dim'] = 10
baseOpts['emg_dim']    = 2


if __name__ == '__main__':
    np.random.seed(42)

    if not os.path.isfile(SAVE_PATH):
        # split hyperparams list into chunks and select chunk that corresponds to this job ID
        sweep_args = sweep_utils.generateArgs(sweepOpts, baseOpts)
        sweep_args = np.array_split(sweep_args, args.n_jobs)[args.jobID]  # hack to fix remaining jobs

        print('*****ADAN sweep*****')
        print('Number of jobs: ', len(sweep_args))
        print('Running...')

        tf.set_random_seed(seed=42)
        tf.disable_eager_execution()


        scores = list()
        for arg in sweep_args:
            scores.append(sweep_utils.test_ADAN(arg, model_folder))

        print('Done.')
        np.save(SAVE_PATH, scores)
    
    
   









