# Script for training ADAN models on single sessions -- see Supp_Fig_1.ipynb for plotting
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import glob, sys
import argparse

[sys.path.append(f) for f in glob.glob('../utils/*')]
from preprocess import *
from plotting_utils import *
from lineplots import plotsd
from stabilizer_utils import *
from recalibration_utils import *
from session_utils import *

from adan_utils import *
from numba import cuda 


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
from tqdm import notebook


device = cuda.get_current_device()
device.reset()

tf.reset_default_graph()
tf.set_random_seed(seed=42)
tf.disable_eager_execution()


# general settings:
min_nblocks = 2
participant = 'T5'
train_size  = 0.5
sigma       = 2
task        = None
save_dir    = '/oak/stanford/groups/shenoy/ghwilson/nonstationarities/T5/ADAN/models/'

# best: normalize : False, offset_strength : 0.25, whitenoise_strength : 0.5, randomwalk_strength : 0.09
cfg = {
    'spike_dim' : 192,
    'emg_dim': 2,
    'latent_dim' : 10,
    'batch_size' : 64,
    'n_steps' : 4,
    'n_layers': 1,
    'n_epochs' : 50,
    'lr' : 0.001,
    'center' : True,
    'normalize' : False,              # True
    'noise_kwargs' : {
        'offset_strength' : 3,        # 0.25
        'whitenoise_strength' : 14,   # 0.5
        'randomwalk_strength' : 0.9   # 0.09
        },
    'save_dir' : None
    }


#----------------------------------------
FILE_DIR       = '/oak/stanford/groups/shenoy/ghwilson/nonstationarities/' + participant + '/'
fig_path       = '/home/users/ghwilson/projects/nonstationarities/figures/'
filelist       = glob.glob(FILE_DIR + 'historical/*')
filelist.extend(glob.glob(FILE_DIR + 'new/*'))

block_constraints = getBlockConstraints(FILE_DIR)
files             = get_Sessions(filelist, min_nblocks,  block_constraints = block_constraints)

parser = argparse.ArgumentParser(description = 'Code for optimizing HMM across session pairs.')
parser.add_argument('--jobID', type = int, help = 'job ID')
args  = parser.parse_args()
    

files  = [files[args.jobID]]
scores = np.zeros((len(files), 2))

for i, file in enumerate(files):
    fields     = ['TX', 'cursorPos', 'targetPos', 'displacement']

    day0        = DataStruct(file, alignScreens = True, causal_filter = sigma)
    ref_dat     = getTrainTest(day0, train_size = train_size, task = task, blocks = block_constraints[file], 
                               fields = fields, returnFlattened = True)
    
    datas    = getADANData(ref_dat, ref_dat, n_steps = cfg['n_steps'], center = cfg['center'], normalize = cfg['normalize'])
    spike_tr = datas['ref_train_TX']
    emg_tr   = datas['ref_train_displacement']
    spike_te = datas['ref_test_TX']
    emg_te   = datas['ref_test_displacement']
    
    cfg['save_dir'] = os.path.join(save_dir, day0.date)
        
    tf.reset_default_graph()
    scores[i, :] = train_network(spike_tr, emg_tr, spike_te, emg_te, cfg)
    
    np.save(os.path.join(save_dir, 'scores', f'scores_{args.jobID}.npy'), scores)
