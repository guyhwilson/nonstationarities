import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import copy, glob, sys, joblib, argparse
from joblib import Parallel, delayed

import os, re
HOME = os.path.expanduser('~')

[sys.path.append(f) for f in glob.glob(HOME + '/projects/nonstationarities/utils/*')]
from preprocess import DataStruct
import preprocess
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
#from adan_utils import *

#import tensorflow.compat.v1 as tf
from numba import cuda
from tqdm import notebook




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
    
    # no recalibration whatsoever
    score_dict['norecal_R2_score']  = pair_data['norecal_R2_score']
    score_dict['norecal_pearson_r'] = pair_data['norecal_pearson_r']
    
    # mean recalibration only 
    score_dict['meanrecal_R2_score'] = pair_data['meanrecal_R2_score']      
    score_dict['meanrecal_pearson_r']= pair_data['meanrecal_pearson_r'] 
    
    # full supervised recalibration 
    score_dict['suprecal_R2_score']  = pair_data['suprecal_R2_score']
    score_dict['suprecal_pearson_r'] = pair_data['suprecal_pearson_r']

    return score_dict


def formatJobOutput(f, prune = None):
    raw = np.load(f, allow_pickle = True)
    df  = pd.DataFrame([x for x in raw])
    
    if prune is not None and not df.empty:
        df = get_subsetDF(df, prune)
    
    return df

def getSummaryDataFrame(files, fields = None, prune = None, int_encode_files = False):
    '''Format list of output files from test_XXX() calls in batchSweep.sh
       Returns as a pandas dataframe. Inputs are:
       
           files (list of str)  - files to process 
           fields (list of str) - rows to subselect
           prune (dict)         - dictionary restricting rows of dataframes 
                                  based on specified values; keys correspond to
                                  column labels, values to desired column value'''
    
    scores = list()
    # avoid doing all at once to avoid overusing memory (slower but no crashes)
    for i, f in enumerate(files):
        formatted = formatJobOutput(f, prune)
        
        if fields is not None and not formatted.empty:
            try:
                formatted = formatted[fields]
            except:
                print('Failure: field not found in ', f)
            
        # if true, encode file field with integer tuple to represent dates 
        # (for cutting down on memory usage by string encoding) 
        columns = formatted.columns.values.tolist()
        if int_encode_files:
            if 'file' not in columns:
                raise ValueError('Column \'files\' not in df but file int encoding requested.')
            else: 
                init = '2016.09.26'
                formatted['file'] = formatted['file'].apply(lambda x: [preprocess.daysBetween(x.split('_to_')[0], init),
                                                                       preprocess.daysBetween(x.split('_to_')[1], init)] )
        
        scores.append(formatted)

    df = pd.concat(scores).reset_index()
    
    if 'days_apart' not in df.columns:
        df['days_apart'] = df.apply(lambda row: get_time_difference(row['file']), axis=1)

    return df


def get_subsetDF(df, query_dict):
    '''Subselect pd dataframe based on arbitrary column values.'''
    
    df_copy = copy.deepcopy(df)
    for key, value in query_dict.items():
        df_copy = df_copy.loc[(df_copy[key] == value)]
    
    return df_copy


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
    
    new_decoder, prob = HMM.recalibrate(decoder, train_neural, train_cursorPos, probThreshold = arg['probWeighted'],
                                       return_viterbi_prob = True)
    score_dict        = makeScoreDict(new_decoder, test_neural, test_targvec, arg, pair_data)
    
    score_dict['viterbi_prob'] = prob
    
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


def test_HMM_Stabilizer_old(arg):
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
    latent_decoder       = LinearRegression(fit_intercept = False, normalize = False).fit(A_train_latent, A_train_targvec)

    # now fit to new day, find mapping, and test mapped data:         
    stab.fit_new(B_train_neural, B = arg['B'], thresh = arg['thresh'], conditionAveraged = False)
    G                  = stab.getNeuralToLatentMap(stab.new_model)
    stabilized_decoder = latent_decoder.coef_.dot(G.dot(stab.R).T) # X --> Z' --> Z ---> Y
    
        
    # %%%%%%%% use subspace-recalibrated decoder's predictions (+ cursor pos) on train B block; pass to HMM %%%%%%%
    
    # make target states - pull screen bounds from pair_data file, get gridSize from args:
    X_min, X_max, Y_min, Y_max = pair_data['B_screenBounds']
    X_loc,Y_loc                = np.meshgrid(np.linspace(X_min, X_max, arg['gridSize']), 
                                             np.linspace(Y_min, Y_max, arg['gridSize']))
    targLocs                   = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T
    
    HMM = HMMRecalibration(arg['stateTrans'], targLocs, arg['pStateStart'], arg['kappa'], 
                                 adjustKappa = lambda dist : 1 / (1 + np.exp(-1 * (dist - arg['inflection']) *arg['exp'])))
       
    decoder           = copy.deepcopy(pair_data['A_decoder'])
    train_neural      = [pair_data['B_train_neural']]
    train_cursorPos   = [pair_data['B_train_cursor']]
    test_neural       = np.vstack(pair_data['B_test_neural'])
    test_targvec      = np.vstack(pair_data['B_test_targvec'])
    
    HMM_decoder = HMM.recalibrate(decoder, train_neural, train_cursorPos, probThreshold = arg['probWeighted'],
                                  return_viterbi_prob = False)
    
    new_decoder            = LinearRegression(fit_intercept = False, normalize = False)
    new_decoder.coef_      = (stabilized_decoder + HMM_decoder.coef_) / 2 
    new_decoder.intercept_ = 0
    score_dict             = makeScoreDict(new_decoder, B_test_neural, test_targvec, arg, pair_data)
    
    return score_dict


def test_ADAN(arg, model_folder):
    '''Test subspace stabilizer using generated session pairs dataset. Input <arg> is 
       dictionary with key-value pairs: '''
    

    # load tf model file (use /session_pairs/ bc pickled files have compatible sklearn models)
    model_date  = re.search(r'(\d+.\d+.\d+)', arg['file']).group(0)
    model_dir   = os.path.join(model_folder, model_date)
    tf.reset_default_graph()

    g       = tf.train.import_meta_graph(os.path.join(model_dir, 'decoder.meta'))
    graph   = tf.get_default_graph()
    spike   = graph.get_tensor_by_name("spike:0")
    emg_hat =  graph.get_tensor_by_name(name="emg_hat:0")

    input_day0 = tf.placeholder(tf.float32, (None, arg['spike_dim']), name='input_day0')
    input_dayk = tf.placeholder(tf.float32, (None, arg['spike_dim']), name='input_dayk')

    # load pickled data
    arg['file'] = arg['file'].replace('/train/', '/session_pairs/')
    pair_data   = np.load(arg['file'], allow_pickle = True).item()
    pair_data.keys()

    spike_day0 = np.concatenate([pair_data['A_train_neural'], pair_data['A_test_neural']])
    spike_dayk = pair_data['B_train_neural']
    spike_test = pair_data['B_test_neural']

    emg_day0 = np.concatenate([pair_data['A_train_targvec'], pair_data['A_test_targvec']])
    emg_dayk = pair_data['B_train_targvec']
    emg_test = pair_data['B_test_targvec']

    
    def generator(input_, reuse=False):
        with tf.variable_scope('generator',initializer=tf.initializers.identity(),reuse=reuse):
            h1 = tf.layers.dense(input_,  arg['spike_dim'], activation=tf.nn.elu)
            output  = tf.layers.dense(h1, arg['spike_dim'], activation=None)
        return output

    def discriminator(input_, n_units=[64,32, arg['latent_dim']], reuse=False):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            noise = tf.random_normal(tf.shape(input_), dtype=tf.float32) * 50
            input_ = input_+noise
            h1 = tf.layers.dense(input_, units=n_units[0], activation=tf.nn.elu)
            h2 = tf.layers.dense(h1, units=n_units[1], activation=tf.nn.elu)
            latent = tf.layers.dense(h2, units=n_units[2], activation=None)
            h3 = tf.layers.dense(latent, units=n_units[1], activation=tf.nn.elu)
            h4 = tf.layers.dense(h3, units=n_units[0], activation=tf.nn.elu)
            logits = tf.layers.dense(h4, units=arg['spike_dim'], activation=None)
            return latent, logits


    # setup generator and discriminator, as well as loss fcns
    input_dayk_aligned      = generator(input_dayk)
    latent_day0,logits_day0 = discriminator(input_day0)
    latent_dayk,logits_dayk = discriminator(input_dayk_aligned)

    d_loss_0 = tf.reduce_mean(tf.abs(logits_day0-input_day0)) 
    d_loss_k = tf.reduce_mean(tf.abs(logits_dayk-input_dayk_aligned))
    d_loss = d_loss_0 - d_loss_k
    g_loss = d_loss_k

    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    d_opt = tf.train.AdamOptimizer(learning_rate=arg['d_lr']).minimize(d_loss, var_list=d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=arg['g_lr']).minimize(g_loss, var_list=g_vars)
    
    
    n_batches  = min(len(spike_day0),len(spike_dayk))//(arg['batch_size'])
    a_vars     = [var.name for var in t_vars if var.name.startswith('autoencoder')]
    for i,name in enumerate(a_vars):
        tf.train.init_from_checkpoint(os.path.join(model_dir, ''), {name[:-2]:d_vars[i]})

    init = tf.global_variables_initializer()

    with tf.Session() as sess: 
        init.run()
        g.restore(sess, tf.train.latest_checkpoint(os.path.join(model_dir, '')))
        for epoch in notebook.tqdm(range(arg['n_epochs'])):
            spike_0_gen_obj = get_batches(spike_day0, arg['batch_size'])
            spike_k_gen_obj = get_batches(spike_dayk, arg['batch_size'])

            for ii in range(n_batches):
                spike_0_batch = next(spike_0_gen_obj)
                spike_k_batch = next(spike_k_gen_obj)
                sys.stdout.flush()
                _,g_loss_ = sess.run([g_opt,g_loss],feed_dict={input_day0:spike_0_batch,input_dayk:spike_k_batch})
                _,d_loss_0_,d_loss_k_ = sess.run([d_opt,d_loss_0,d_loss_k],feed_dict={input_day0:spike_0_batch,input_dayk:spike_k_batch})

            if (epoch % 10 == 0) or (epoch == arg['n_epochs']-1):
                print("\r{}".format(epoch), "Discriminator loss_day_0:",d_loss_0_,"\Discriminator loss_day_k:",d_loss_k_)
                input_dayk_aligned_ = input_dayk_aligned.eval(feed_dict={input_dayk:spike_dayk})
                emg_dayk_aligned    = emg_hat.eval(feed_dict={spike:input_dayk_aligned_})
                emg_k_              = emg_hat.eval(feed_dict={spike:spike_dayk})
                print("EMG non-aligned VAF:", vaf(emg_dayk,emg_k_),
                      "\tEMG aligned VAF:", vaf(emg_dayk,emg_dayk_aligned),
                     "\tEMG aligned r: ", np.corrcoef(emg_dayk.flatten(), emg_dayk_aligned.flatten())[0, 1])
                
        input_test_aligned_ = input_dayk_aligned.eval(feed_dict={input_dayk:spike_test})
        emg_test_aligned    = emg_hat.eval(feed_dict={spike:input_test_aligned_})
                  
    score_dict = dict(arg)
    score_dict['R2_score']   = sklearn.metrics.r2_score(emg_test,emg_test_aligned)                         
    score_dict['pearson_r']  = np.corrcoef(emg_test.flatten(), emg_test_aligned.flatten())[0, 1]                    
    score_dict['days_apart'] = pair_data['days_apart']
    
    # no recalibration whatsoever
    score_dict['norecal_R2_score']  = pair_data['norecal_R2_score']
    score_dict['norecal_pearson_r'] = pair_data['norecal_pearson_r']
    
    # mean recalibration only 
    score_dict['meanrecal_R2_score'] = pair_data['meanrecal_R2_score']      
    score_dict['meanrecal_pearson_r']= pair_data['meanrecal_pearson_r'] 
    
    # full supervised recalibration 
    score_dict['suprecal_R2_score']  = pair_data['suprecal_R2_score']
    score_dict['suprecal_pearson_r'] = pair_data['suprecal_pearson_r']

    return score_dict



