import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import sys, glob, os
from utils.plotting.plotting_utils import figSize
from utils.simulation.simulation import simulateBCIFitts
from utils.simulation import simulation_utils
from utils.preprocessing import sweep_utils

from utils.recalibration import RTI_utils, stabilizer_utils, hmm_utils, hmm
from stabilizer_utils import Stabilizer
from hmm import HMMRecalibration
from RTI_utils import RTI

import copy
from joblib import Parallel, delayed
import argparse





def initializeMethod(base_opts, method_opts):

    
    if method_opts['method'] == 'stabilizer':
        cfg, ss_decoder_dict, stabilizer = simulation_utils.initializeBCI({**base_opts, **method_opts})
        method = (ss_decoder_dict, stabilizer, method_opts)
        
    elif method_opts['method'] == 'RTI':
        cfg    = simulation_utils.initializeBCI(base_opts)
        method = RTI(method_opts['look_back'], method_opts['min_dist'], method_opts['min_time'])

    elif method_opts['method'] == 'PRI-T':
        cfg                     = simulation_utils.initializeBCI(base_opts)
        adjustKappa             = lambda x: 1 / (1 + np.exp(-1 * (x - method_opts['inflection']) * method_opts['exp']))
        targLocs                = hmm_utils.generateTargetGrid(gridSize = method_opts['gridSize'], is_simulated=True)
        stateTrans, pStateStart = hmm_utils.generateTransitionMatrix(gridSize = method_opts['gridSize'],
                                                                     stayProb = method_opts['stayProb'])

        if method_opts['click_inflection'] is None:
            clickModel = None
        else:
            clickModel = lambda x: 1- (1 / (1 + np.exp(-1 * (x - method_opts['click_inflection'])*12))) 
            
        method = HMMRecalibration(stateTrans, targLocs, pStateStart, method_opts['vmKappa'], 
                                          adjustKappa = adjustKappa, getClickProb = clickModel)
        
    elif method_opts['method'] == 'supervised':
        cfg    = simulation_utils.initializeBCI(base_opts)
        method = []
        
    elif method_opts['method'] == 'gain':
        cfg    = simulation_utils.initializeBCI(base_opts)
        method = []
        
    else:
        raise ValueError('Method not recognized')
        
            
    return cfg, method


def performMethodRecal(cfg, method_opts, method):
    
    if method_opts['method'] == 'stabilizer':
        D = simulation_utils.simulate_LatentClosedLoopRecalibration(cfg, *method)
        
    elif method_opts['method'] == 'RTI':
        D = simulation_utils.simulate_RTIRecalibration(cfg, method)
        
    elif method_opts['method'] == 'PRI-T':
        D = simulation_utils.simulate_HMMRecalibration(cfg, method)
        
    elif method_opts['method'] == 'supervised':
        cfg_copy      = copy.deepcopy(cfg)
        cfg_copy['D'] = simulation_utils.simulate_OpenLoopRecalibration(cfg_copy, nSteps = 10000)
        D             = simulation_utils.simulate_ClosedLoopRecalibration(cfg_copy)
        
    elif method_opts['method'] == 'gain':
        D = cfg['D']
        
    else:
        raise ValueError('Method not recognized')

        
    return D
    


def testMethod(base_opts, method_opts, save_fields = None):
    
    scores_dict = copy.deepcopy(method_opts)
    scores      = np.zeros((base_opts['n_sessions']))
    
    cfg, method = initializeMethod(base_opts, method_opts)
    

    for j in range(base_opts['n_sessions']):
        new_SNR             = base_opts['SNR'] if base_opts['fixed_SNR'] else simulation_utils.sampleSNR()
        cfg['neuralTuning'] = simulation_utils.simulateTuningShift(cfg['neuralTuning'], n_stable = base_opts['n_stable'], 
                                                                   PD_shrinkage = base_opts['shrinkage'], mean_shift = 0, 
                                                                   renormalize = new_SNR)  
        
        test_cfg         = copy.deepcopy(cfg)
        test_cfg['D']    = performMethodRecal(test_cfg, method_opts, method)
        test_cfg['beta'] = simulation_utils.gainSweep(test_cfg, possibleGain = base_opts['possibleGain'])

        scores[j] = simulation_utils.evalOnTestBlock(test_cfg)
    
        if method_opts['chained'] or j == base_opts['n_sessions']-1:
            cfg['D']    = test_cfg['D']
            cfg['beta'] = test_cfg['beta']

    scores_dict['ttt'] = scores
    
    if save_fields is not None:
        for field in save_fields:
            scores_dict[field] = cfg[field]

    return scores_dict



# ============================================================
#np.random.seed(42)

# general settings:
reps  = 200   # how many times to repeat the repeated nonstationarities simulation

base_opts = dict()
base_opts['alpha']          = 0.94 # amount of exponential smoothing (0.9 to 0.96 are reasonable)
base_opts['delT']           = 0.02 # define the time step (20 ms)
base_opts['nDelaySteps']    = 10   # define the simulated user's visual feedback delay (200 ms)
base_opts['nSimSteps']      = 20000
base_opts['nUnits']         = 192
base_opts['SNR']            = 0.5
base_opts['fixed_SNR']      = False
base_opts['possibleGain']   = np.linspace(0.1,2.5,10)
base_opts['center_means']   = True
base_opts['nTrainingSteps'] = 10000

base_opts['n_sessions']   = 60   # number of sessions to simulate 
base_opts['days_between'] = 0    # days between session days
base_opts['shrinkage']    = 0.91  # relative tuning in subspace per new day
base_opts['n_stable']     = 0


# stabilizer settings:
ss_opts = {
    'method' : 'stabilizer',
    'B'      : 190,
    'thresh' : 0.05,
    'n_components' : 4,
    'model_type'   : 'PCA',
    'chained'      : True
} 

# static stabilizer settings: 
static_ss_opts = {
    'method' : 'stabilizer',
    'B'      : 130,
    'thresh' : 0.05,
    'n_components' : 2,
    'model_type'   : 'PCA',
    'chained'      : False
} 



# PRI-T settings:
hmm_opts = {
    'method'      : 'PRI-T',
    'probThresh'  : 'probWeighted',
    'gridSize'    : 20,
    'stayProb'    : 0.999,
    'inflection'  : 0.2,
    'exp'         : 1,
    'vmKappa'     : 4,
    'chained'     : True,
    'click_inflection': None
}

# static PRI-T settings:
static_hmm_opts = {
    'method'      : 'PRI-T',
    'probThresh'  : 'probWeighted',
    'gridSize'    : 20,
    'stayProb'    : 0.999,
    'inflection'  : 0.3,
    'exp'         : 8.8,
    'vmKappa'     : 4,
    'chained'     : False,
    'click_inflection': None
}


# RTI settings:
rti_opts = {
    'method'    : 'RTI',
    'look_back' : 320,
    'min_dist'  : 0.1,
    'min_time'  : 30,
    'chained'   : True
}

# static RTI settings:
static_rti_opts = {
    'method'    : 'RTI',
    'look_back' : 400,
    'min_dist'  : 0.1,
    'min_time'  : 10,
    'chained'   : False
}


gain_opts = {'method' : 'gain',
             'chained' : False}

supervised_opts = {'method' : 'supervised',
                   'chained' : True}
            


# ============================================================


parser = argparse.ArgumentParser(description = 'Code for measuring performance across time.')
parser.add_argument('--n_jobs', type = int, help = 'Number of jobs running this script')
parser.add_argument('--jobID', type = int, help = 'job ID')
parser.add_argument('--saveDir', type = str, default = './', help = 'Folder for saving scores')

if __name__ == '__main__':
    
    args  = parser.parse_args()
    
    sweep_args = [hmm_opts, static_hmm_opts, ss_opts, static_ss_opts, rti_opts, 
                  static_rti_opts, gain_opts, supervised_opts] * reps
    job_args   = np.array_split(sweep_args, args.n_jobs)[args.jobID]
    
    if not os.path.isdir(args.saveDir):
        os.makedirs(args.saveDir)
        
    save_fname = os.path.join(args.saveDir, f'sweep_scores_{args.jobID}.npy')
    
    print('Number of runs for this job: ', len(job_args))
    print('File save path:', save_fname)
    
    rep_data = Parallel(n_jobs= -1, verbose = 11)(delayed(testMethod)(base_opts, x) for x in job_args)
    np.save(save_fname, rep_data)
    print('Finished.')
    
    #sweep_args = [[base_opts, hmm_opts, static_hmm_opts, ss_opts, static_ss_opts, rti_opts, static_rti_opts]] * reps
    #job_args   = np.array_split(sweep_args, args.n_jobs)[args.jobID]
    #save_fname = os.path.join(args.saveDir, f'sweep_scores_{args.jobID}.npy')
    
    #print('Number of runs for this job: ', len(job_args))
    #print('File save path:', save_fname)
    
    #rep_data = Parallel(n_jobs= -1, verbose = 11)(delayed(simulate_MultiSessionStretch)(*x) for x in job_args)
    #np.save(save_fname, rep_data)
    #print('Finished.')




