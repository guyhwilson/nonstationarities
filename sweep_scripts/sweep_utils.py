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

        
def test_Stabilizer(arg):
    '''Test subspace stabilizer using generated session pairs dataset. Input <arg> is 
       dictionary with key-value pairs:

        'model'
        'n_components'
        'B'
        'thresh'
        'A_[train/test]_neural'
        'A_[train/test]_targvec'
        '''
    pair_data = np.load(arg['file'], allow_pickle = True).item()
    
    # data for building latent decoder
    A_train_neural      = pair_data['A_train_neural']
    A_train_targvec     = pair_data['A_train_targvec']
    
    # data for performing realignment
    B_train_neural      = pair_data['B_train_neural']
    B_train_targvec     = pair_data['B_train_targvec']
    
    # data for testing
    B_test_neural       = np.vstack(pair_data['B_test_neural'])
    B_test_targvec      = np.vstack(pair_data['B_test_targvec'])
    
    # fit dimensionality reduction method to train latent decoder:
    stab                 = Stabilizer(arg['model'], arg['n_components'])
    stab.fit_ref(A_train_neural, conditionAveraged = False)
    A_train_latent       = stab.ref_model.transform(A_train_neural)
    latent_decoder       =  LinearRegression(normalize = False).fit(A_train_latent, A_train_targvec)

    # now fit to new day, find mapping, and test mapped data:         
    stab.fit_new(B_train_neural, B = arg['B'], thresh = arg['thresh'], conditionAveraged = False)
    B_test_latent  = stab.transform(B_test_neural)
    score          = latent_decoder.score(B_test_latent, B_test_targvec)

    return score