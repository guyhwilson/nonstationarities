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
base_opts['center_means']   = False
base_opts['nTrainingSteps'] = 10000


base_opts['n_sessions']   = 30     # number of sessions to simulate 
base_opts['days_between'] = 0      # days between session days
base_opts['shrinkage']    = 0.91   # relative tuning in subspace per new day
base_opts['n_stable']     = 0

# sweep settings:
sweep_opts = dict()
sweep_opts['nSimSteps'] = [2000, 3000, 4000, 5000, 6000]


# HMM settings:
vmKappa    = 2              # Precision parameter for the von mises distribution
probThresh = 'probWeighted' 
gridSize   = 20
stayProb   = 0.999
adjustKappa = lambda x: 1 / (1 + np.exp(-1 * (x - 0.) * 32.))
clickModel  = lambda x:  (x < 0.075).astype(int)


# RTI settings:
look_back = 240
min_dist  = 0
min_time  = 30


##############################


def testReducedData(cfg, test_n_timestamps = 20000):
    
    cfg_copy              = copy.deepcopy(cfg)
    cfg_copy['nSimSteps'] = test_n_timestamps
    
    ttt = np.mean(simulateBCIFitts(cfg_copy)['ttt']) 
    
    return ttt


def simulate_MultiSessionStretch(base_opts, hmm, clickhmm, rti):
    
    session_scores = np.zeros((base_opts['n_sessions'] + 1, 4))
    
    # set up each case and run initial performance eval:
    cfg_dict   = dict()
    order_dict = {'supervised_cfg' : 0}
    cfg_dict['supervised_cfg'] = simulation_utils.initializeBCI(base_opts)
    cfg_dict['supervised_cfg']['nSimSteps'] /= 2
    session_scores[0, 0] = np.mean(simulateBCIFitts(cfg_dict['supervised_cfg'])['ttt'])
    
    cfg_dict['hmm_cfg']  = simulation_utils.initializeBCI(base_opts)
    session_scores[0, 1] = np.mean(simulateBCIFitts(cfg_dict['hmm_cfg'])['ttt'])

    cfg_dict['clickhmm_cfg'] = simulation_utils.initializeBCI(base_opts)
    session_scores[0, 2]     = np.mean(simulateBCIFitts(cfg_dict['clickhmm_cfg'])['ttt'])
    
    cfg_dict['rti_cfg']   = simulation_utils.initializeBCI(base_opts)
    session_scores[0, 3]  = np.mean(simulateBCIFitts(cfg_dict['rti_cfg'])['ttt'])
    
    
    for i in range(base_opts['n_sessions']):
        for j in range(base_opts['days_between'] + 1):
            for cfg in cfg_dict.values():
                # introduce daily nonstationarities between recorded sessions
                cfg['neuralTuning'] = simulation_utils.simulateTuningShift(cfg['neuralTuning'], 
                                                                           n_stable = base_opts['n_stable'],
                                                                           PD_shrinkage = base_opts['shrinkage'], 
                                                                           mean_shift = 0, 
                                                                           renormalize = simulation_utils.sampleSNR())  
        
        
        # supervised CL recal: 
        cfg_dict['supervised_cfg']['D'] = simulation_utils.simulate_OpenLoopRecalibration(cfg_dict['supervised_cfg'])
        cfg_dict['supervised_cfg']['D'] = simulation_utils.simulate_ClosedLoopRecalibration(cfg_dict['supervised_cfg'])
        cfg_dict['supervised_cfg']['beta'] = simulation_utils.gainSweep(cfg_dict['supervised_cfg'], 
                                                                       possibleGain = base_opts['possibleGain'])
        session_scores[i+1, 0]        = testReducedData(cfg_dict['supervised_cfg'])
                
        # vanilla HMM 
        cfg_dict['hmm_cfg']['D']    = simulation_utils.simulate_HMMRecalibration(cfg_dict['hmm_cfg'], hmm)
        cfg_dict['hmm_cfg']['beta'] = simulation_utils.gainSweep(cfg_dict['hmm_cfg'], possibleGain = base_opts['possibleGain'])
        session_scores[i+1, 1]      = testReducedData(cfg_dict['hmm_cfg'])         

        # click HMM 
        cfg_dict['clickhmm_cfg']['D']    = simulation_utils.simulate_HMMRecalibration(cfg_dict['clickhmm_cfg'], clickhmm)
        cfg_dict['clickhmm_cfg']['beta'] = simulation_utils.gainSweep(cfg_dict['clickhmm_cfg'], possibleGain = base_opts['possibleGain'])
        session_scores[i+1, 2]    = testReducedData(cfg_dict['clickhmm_cfg'])
        
        # RTI
        cfg_dict['rti_cfg']['D']    = simulation_utils.simulate_RTIRecalibration(cfg_dict['rti_cfg'], rti)
        cfg_dict['rti_cfg']['beta'] = simulation_utils.gainSweep(cfg_dict['rti_cfg'], possibleGain = base_opts['possibleGain'])
        session_scores[i+1, 3]      = testReducedData(cfg_dict['rti_cfg'])
        
        
    scores_dict = dict()
    scores_dict['scores']    = session_scores
    scores_dict['nSimSteps'] = base_opts['nSimSteps']

    return scores_dict



if __name__ == '__main__':
    

    session_scores          = np.zeros((reps, base_opts['n_sessions'] + 1, 4)) 
    targLocs                = hmm_utils.generateTargetGrid(gridSize = gridSize)
    stateTrans, pStateStart = hmm_utils.generateTransitionMatrix(gridSize = gridSize, stayProb = stayProb)

    hmm                     = HMMRecalibration(stateTrans, targLocs, pStateStart, vmKappa, adjustKappa = adjustKappa)
    clickhmm                = HMMRecalibration(stateTrans, targLocs, pStateStart, vmKappa, adjustKappa = adjustKappa,
                                               getClickProb = clickModel)
    rti                     = RTI(look_back, min_dist, min_time)
    
    sweep_opts = sweep_utils.generateArgs(sweep_opts, base_opts)
    sweep_opts = [[x, hmm, clickhmm, rti] for x in sweep_opts] * reps
    sweep_opts = np.array_split(sweep_opts, args.n_jobs)[args.jobID]
    
    sweep_scores = Parallel(n_jobs= -1, verbose = 11)(delayed(simulate_MultiSessionStretch)(*x) for x in sweep_opts)
    np.save(args.saveDir + 'efficiency_scores_{}.npy'.format(args.jobID), sweep_scores)
    print('Finished.')

    
    
    
    
    
