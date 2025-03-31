import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import sys, glob
import argparse
import copy
from joblib import Parallel, delayed


[sys.path.append(f) for f in glob.glob('../utils/*')]
from hmm import HMMRecalibration
import hmm_utils
from RTI_utils import RTI
import simulation_utils 
from simulation import simulateBCIFitts
import sweep_utils

from performance_sweep import *
from itertools import product

parser = argparse.ArgumentParser(description = 'Code for optimizing HMM across session pairs.')
parser.add_argument('--n_jobs', type = int, help = 'Number of jobs running this script')
parser.add_argument('--jobID', type = int, help = 'job ID')
parser.add_argument('--saveDir', type = str, default = './', help = 'Folder for saving scores')
args  = parser.parse_args()


##############################
# for a reproducible result
#np.random.seed(1)

# general settings:
reps       = 200   # how many times to repeat for each nSteps sweep

base_opts = dict()
base_opts['alpha']          = 0.94 # amount of exponential smoothing (0.9 to 0.96 are reasonable)
base_opts['delT']           = 0.02 # define the time step (20 ms)
base_opts['nDelaySteps']    = 10   # define the simulated user's visual feedback delay (200 ms)
base_opts['nUnits']         = 192
base_opts['possibleGain']   = np.linspace(0.1,2.5,10)
base_opts['center_means']   = True
base_opts['nTrainingSteps'] = 10000
base_opts['nSimSteps']      = 20000


base_opts['n_sessions']   = 30     # number of sessions to simulate 
base_opts['days_between'] = 0      # days between session days
base_opts['shrinkage']    = 0.91   # relative tuning in subspace per new day
base_opts['n_stable']     = 0
base_opts['fixed_SNR']    = True

# sweep settings:
sweep_opts = dict()
sweep_opts['SNR'] = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]



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
    'click_inflection' : None
    
}

# Click PRI-T settings:
click_hmm_opts = {
    'method'      : 'PRI-T',
    'probThresh'  : 'probWeighted',
    'gridSize'    : 20,
    'stayProb'    : 0.999,
    'inflection'  : 0.2,
    'exp'         : 1,
    'vmKappa'     : 4,
    'chained'     : True,
    'click_inflection' : 0.1
}


# RTI settings:
rti_opts = {
    'method'    : 'RTI',
    'look_back' : 320,
    'min_dist'  : 0.1,
    'min_time'  : 30,
    'chained'   : True
}


# long lookback RTI settings:
long_rti_opts = {
    'method'    : 'RTI',
    'look_back' : 500,
    'min_dist'  : 0.1,
    'min_time'  : 30,
    'chained'   : True
}

# stabilizer settings:
ss_opts = {
    'method' : 'stabilizer',
    'B'      : 190,
    'thresh' : 0.05,
    'n_components' : 4,
    'model_type'   : 'PCA',
    'chained'      : True
} 


supervised_opts = {'method' : 'supervised',
                   'chained' : True}


##############################


if __name__ == '__main__':
  
    base_opts_list = sweep_utils.generateArgs(sweep_opts, base_opts)
    method_opts    = [hmm_opts, click_hmm_opts, rti_opts, long_rti_opts, ss_opts, supervised_opts] * reps
    
    sweep_opts = list(product(base_opts_list, method_opts))
    print(len(sweep_opts), 'total jobs')
    
    sweep_opts = np.array_split(sweep_opts, args.n_jobs)[args.jobID]
    print(len(sweep_opts), 'jobs for this run')
    
    if not os.path.isdir(args.saveDir):
        os.makedirs(args.saveDir)

    sweep_scores = Parallel(n_jobs= -1, verbose = 11)(delayed(testMethod)(*x, save_fields = ['neuralTuning']) for x in sweep_opts)
        
    np.save(os.path.join(args.saveDir, f'SNR_scores_{args.jobID}.npy'), sweep_scores)
    print('Finished.')

