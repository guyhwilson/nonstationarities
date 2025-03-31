import numpy as np
import sys, glob, copy, os

from utils.plotting.plotting_utils import figSize
from utils.simulation.simulation import simulateBCIFitts
from utils.simulation import simulation_utils
from utils.preprocessing import sweep_utils

from utils.recalibration import RTI_utils, stabilizer_utils, hmm_utils, hmm
from stabilizer_utils import Stabilizer
from hmm import HMMRecalibration
from RTI_utils import RTI

from joblib import Parallel, delayed
import copy

from sklearn.linear_model import LinearRegression

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from scipy.linalg import orthogonal_procrustes
from utils.PD_tools.CosineTuning import getAngles
import argparse

from performance_sweep import *



def getProcrustesError(cfg, stabilizer, ss_decoder_dict):
    neural, kinematics = simulation_utils.simulateUnitActivity(cfg['neuralTuning'], noise=0.3,
                                                               nSteps=cfg['nSimSteps'])

    latent = stabilizer.new_model.transform(neural)
    
    kinematics_mapped_to_reference = kinematics.dot(np.linalg.inv(ss_decoder_dict['h'].T))
    R_sup, _             = orthogonal_procrustes(latent, kinematics_mapped_to_reference)
        
    angles = getAngles(R_sup, stabilizer.R, returnFullAngle=False) * 180 / np.pi
    
    return angles


def getAngularError(encoder, decoder):
    return np.arccos(np.corrcoef(encoder.flatten(), decoder.flatten())[0,1]) * 180/np.pi


def getUnstableDays(base_opts, ss_opts, threshold):
    
    scores_dict = copy.deepcopy(ss_opts)
    
    fields = ['reference_cfgs', 'new_cfgs', 'stabilizers', 'ss_decoder_dicts', 'method', 'ttt', 
              'n_days_to_threshold', 'corrvals']

    for field in fields:
        scores_dict[field] = list()
    
    cfg, method = initializeMethod(base_opts, ss_opts)

    ref_tuning = np.copy(cfg['neuralTuning'])
    scores_dict['reference_cfgs'].append(copy.deepcopy(cfg))

    n_days  = 0
    corrval = 0 
    while corrval < threshold:
        n_days += 1
        cfg['neuralTuning'] = simulation_utils.simulateTuningShift(cfg['neuralTuning'], n_stable = base_opts['n_stable'],
                                                                   PD_shrinkage = base_opts['shrinkage'], 
                                                                      mean_shift = 0, 
                                                                   renormalize = base_opts['SNR'])
        
        cfg['D']    = performMethodRecal(cfg, ss_opts, method)
        cfg['beta'] = simulation_utils.gainSweep(cfg, possibleGain = base_opts['possibleGain'])

        corrval = getAngularError(cfg['D'][1:, :], cfg['neuralTuning'][:, 1:])
        #corrval = np.mean(getProcrustesError(cfg, method[1], method[0]))
        
    scores_dict['ttt'].append(simulateBCIFitts(cfg)['ttt'])
    scores_dict['new_cfgs'].append(copy.deepcopy(cfg))
    scores_dict['stabilizers'].append(method[1])
    scores_dict['ss_decoder_dicts'].append(method[0])
    scores_dict['n_days_to_threshold'].append(n_days)
    scores_dict['corrvals'].append(corrval)

    return scores_dict


# for a reproducible result
np.random.seed(1)


##############################

ss_opts = {
    'method' : 'stabilizer',
    'B'      : 190,
    'thresh' : 0.05,
    'n_components' : 4,
    'model_type'   : 'PCA',
    'chained'      : True
} 


base_opts      = dict()  # unchanging parameters
base_opts['alpha']          = 0.94 # amount of exponential smoothing (0.9 to 0.96 are reasonable)
base_opts['delT']           = 0.02 # define the time step (20 ms)
base_opts['nDelaySteps']    = 10   # define the simulated user's visual feedback delay (200 ms)
base_opts['nSimSteps']      = 10000
base_opts['nUnits']         = 192
base_opts['possibleGain']   = np.linspace(0.1,2.5,10)
base_opts['center_means']   = True
base_opts['nTrainingSteps'] = 10000
base_opts['shrinkage']      = 0.91
base_opts['n_stable']       = 0
base_opts['SNR']            = 0.5 # we keep SNR fixed to this to remove variability in this analysis


base_opts['model_type']    = 'PCA'


nReps     = 100
threshold = 70

##############################


parser = argparse.ArgumentParser(description = 'Code for measuring performance across time.')
parser.add_argument('--n_jobs', type = int, help = 'Number of jobs running this script')
parser.add_argument('--jobID', type = int, help = 'job ID')
parser.add_argument('--saveDir', type = str, default = './', help = 'Folder for saving scores')
args  = parser.parse_args()

if __name__ == '__main__':
    
    # This will take a few hours to run (sorry!) - you can try to parallelize it for faster inference
    sweep_opts = np.array_split([ss_opts] * nReps, args.n_jobs)[args.jobID]

    print('Number of jobs for process:', len(sweep_opts))
    sweep_scores =  Parallel(n_jobs= -1, verbose = 11)(delayed(getUnstableDays)(base_opts, x, threshold) for x in sweep_opts)
        
    if not os.path.isdir(args.saveDir):
        os.makedirs(args.saveDir)

    np.save(os.path.join(args.saveDir, f'unstable_data_{args.jobID}.npy'), sweep_scores)
    print('Finished.')
    
    
 




