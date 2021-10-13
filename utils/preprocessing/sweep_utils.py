import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
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

import sklearn, scipy 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.model_selection import ParameterGrid

from hmm_utils import HMMRecalibration
from stabilizer_utils import *




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


def makeScoreDict(decoder, test_x, test_y, arg, pair_data):
    '''Helper function that generates output dictionary for all 
       test_XXX() functions. '''
    
    
    pred      = decoder.predict(test_x)
    r2_score  = sklearn.metrics.r2_score(test_y, pred)
    pearson_r = scipy.stats.pearsonr(test_y.flatten(), pred.flatten())[0]
    
    
    score_dict = dict(arg)
    score_dict['R2_score']           = r2_score                         # record model R2
    score_dict['pearson_r']          = pearson_r                        # and pearson r (in case of just gain difference)
    score_dict['days_apart']         = pair_data['days_apart']          # days between sessions
    score_dict['meanRecal_R2_score'] = pair_data['mean_recal_R2']       # mean recalibration R2
    score_dict['meanRecal_pearson_r']= pair_data['mean_recal_pearsonr'] # mean recalibration pearson r
    
    return score_dict


def formatJobOutput(f):
    raw = np.load(f, allow_pickle = True)
    df  = pd.DataFrame([x for x in raw])
    
    return df

def getSummaryDataFrame(files, fields = None):
    '''Format list of output files from test_XXX() calls in batchSweep.sh
       Returns as a pandas dataframe. '''
    
    scores = [formatJobOutput(f) for f in files]
    df     = pd.concat(scores)
    
    if 'days_apart' not in df.columns:
        df['days_apart'] = df.apply(lambda row: get_time_difference(row['file']), axis=1)
    if fields is not None:
        df = df[fields]
    return df


def get_subsetDF(df, query_dict):
    '''Subselect pd dataframe based on arbitrary column values.'''
    
    df = copy.deepcopy(df)
    for key, value in query_dict.items():
        df = df.loc[(df[key] == value)]
    
    return df


def makeStripPlot(df, opt_dict, sweep_dict, var):
    
    opt_copy = dict(opt_dict)
    opt_copy.pop(var)
    
    opt_idx = np.where(sweep_dict[var] == opt_dict[var])[0][0]
    
    sns_arr          = get_subsetDF(df, opt_copy)
    palette          = ['k'] * len(sweep_dict[var])
    palette[opt_idx] = 'r'

    sns.stripplot(data = sns_arr, x = var, y = 'R2_score', hue = var, 
                  palette = palette, orient = 'v', alpha = 0.6)
    plt.legend([], [], frameon = False)

    plt.ylim([-1, 1])
    plt.xlabel(var, fontsize = 12)
    plt.ylabel('$R^2$', fontsize = 12)
    plt.title('Varying ' + var)





def test_HMM(arg):
    '''Test HMM using generated session pairs dataset. Input <arg> is 
       dictionary with key-value pairs:

        'file'         : (str)          - path to session pair data to load
        'probWeighted' : (float or str) - probability threshold or 'probWeighted'
        'pStateStart'  : (2D float)     - nStates x 1 of prior target probabilities
        'stateTrans'   : (2D float)     - state transition matrix 
        'kappa'
        'inflection'
        'exp'         '''
   
    pair_data = np.load(arg['file'], allow_pickle = True).item()
    
    # make target states - pull screen bounds from pair_data file, get gridSize from args:
    X_min, X_max, Y_min, Y_max = pair_data['B_screenBounds']
    X_loc,Y_loc                = np.meshgrid(np.linspace(X_min, X_max, arg['gridSize']), np.linspace(Y_min, Y_max, arg['gridSize']))
    targLocs                   = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T
    
    # define HMM 
    HMM = HMMRecalibration(arg['stateTrans'], targLocs, arg['pStateStart'], arg['kappa'], 
                                 adjustKappa = lambda dist : 1 / (1 + np.exp(-1 * (dist - arg['inflection']) *arg['exp'])))
    
    decoder           = copy.deepcopy(pair_data['A_decoder'])
    train_neural      = [pair_data['B_train_neural']]
    train_cursorPos   = [pair_data['B_train_cursor']]
    test_neural       = np.vstack(pair_data['B_test_neural'])
    test_targvec      = np.vstack(pair_data['B_test_targvec'])
    
    new_decoder = HMM.recalibrate(decoder, train_neural, train_cursorPos)
    score_dict  = makeScoreDict(new_decoder, test_neural, test_targvec, arg, pair_data)
    
    return score_dict

        
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
    score_dict     = makeScoreDict(latent_decoder, B_test_latent, B_test_targvec, arg, pair_data)
    
    return score_dict


def test_HMM_Stabilizer(arg):
    '''Test subspace stabilizer using generated session pairs dataset. Input <arg> is 
       dictionary with key-value pairs:
        '''
    
    
    #%%%%%%%%% recalibrate decoder using subspace realignment %%%%%%%%%%%%%%%
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
    B_train_latent = stab.transform(B_train_neural)
    B_test_latent  = stab.transform(B_test_neural)
    
    # %%%%%%%% use subspace-recalibrated decoder's predictions (+ cursor pos) on train B block; pass to HMM %%%%%%%
    
    # make target states - pull screen bounds from pair_data file, get gridSize from args:
    X_min, X_max, Y_min, Y_max = pair_data['B_screenBounds']
    X_loc,Y_loc                = np.meshgrid(np.linspace(X_min, X_max, arg['gridSize']), np.linspace(Y_min, Y_max, arg['gridSize']))
    targLocs                   = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T
    
    HMM = HMMRecalibration(arg['stateTrans'], targLocs, arg['pStateStart'], arg['kappa'], 
                                 adjustKappa = lambda dist : 1 / (1 + np.exp(-1 * (dist - arg['inflection']) *arg['exp'])))
    
    decoder           = copy.deepcopy(latent_decoder)
    train_cursorPos   = pair_data['B_train_cursor']
    test_targvec      = np.vstack(pair_data['B_test_targvec'])
        
    new_decoder = HMM.recalibrate(decoder, [B_train_latent], [train_cursorPos])
    score_dict  = makeScoreDict(new_decoder, B_test_latent, test_targvec, arg, pair_data)
    
    return score_dict


