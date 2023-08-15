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



def simulate_MultiSessionStretch(base_opts, hmm_opts = None, static_hmm_opts = None,
                                 ss_opts = None, static_ss_opts = None, 
                                 rti_opts = None, static_rti_opts = None):
    '''Simulate multiday cursor control stretches with constantly changing neural activity and differing 
    recalibration approaches. Inputs are dictionaries with key-value pairs:
    
    
        base_opts:
            XX - XX 
            XX - XX
            
        hmm_opts/static_hmm_opts:
        
        ss_opts/static_ss_opts:
        
        rti_opts/static_rti_opts: 
            'look_back' : (int)   - 
            'min_dist'  : (float) -
            'min_time'  : (int)   -
    '''
    
    cfg_dict   = dict()
    order_dict = {'norecal_cfg' : 0, 'supervised_cfg' : 1}
    cfg_dict['supervised_cfg'] = simulation_utils.initializeBCI(base_opts)
    cfg_dict['norecal_cfg']    = simulation_utils.initializeBCI(base_opts)
    
    if hmm_opts is not None:
        targLocs                = hmm_utils.generateTargetGrid(gridSize = hmm_opts['gridSize'], is_simulated=True)
        stateTrans, pStateStart = hmm_utils.generateTransitionMatrix(gridSize = hmm_opts['gridSize'], 
                                                                     stayProb = hmm_opts['stayProb'])
        hmm                     = HMMRecalibration(stateTrans, targLocs, pStateStart, hmm_opts['vmKappa'],
                                                   adjustKappa = hmm_opts['adjustKappa'])
        
        cfg_dict['hmm_cfg']   = simulation_utils.initializeBCI(base_opts)
        order_dict['hmm_cfg'] = len(order_dict.keys())
        
    if static_hmm_opts is not None:
        targLocs                = hmm_utils.generateTargetGrid(gridSize = static_hmm_opts['gridSize'], is_simulated=True)
        stateTrans, pStateStart = hmm_utils.generateTransitionMatrix(gridSize = static_hmm_opts['gridSize'], 
                                                                     stayProb = static_hmm_opts['stayProb'])
        static_hmm              = HMMRecalibration(stateTrans, targLocs, pStateStart, static_hmm_opts['vmKappa'],
                                                   adjustKappa = static_hmm_opts['adjustKappa'])
        
        cfg_dict['static_hmm_cfg']   = simulation_utils.initializeBCI(base_opts)
        order_dict['static_hmm_cfg'] = len(order_dict.keys()) 
        
        
    if ss_opts is not None:
        cfg_dict['ss_cfg'], ss_decoder_dict, stabilizer = simulation_utils.initializeBCI({**base_opts, **ss_opts})
        cfg_dict['ss_cfg']['neuralTuning'][:, 0] = 0
        order_dict['ss_cfg'] = len(order_dict.keys())
        
    if static_ss_opts is not None:
        cfg_dict['static_ss_cfg'], static_ss_decoder_dict, static_stabilizer = simulation_utils.initializeBCI({**base_opts, **static_ss_opts})
        cfg_dict['static_ss_cfg']['neuralTuning'][:, 0] = 0
        order_dict['static_ss_cfg'] = len(order_dict.keys())
        
    if rti_opts is not None:
        rti                   = RTI(rti_opts['look_back'], rti_opts['min_dist'], rti_opts['min_time'])
        cfg_dict['rti_cfg']   = simulation_utils.initializeBCI(base_opts)
        order_dict['rti_cfg'] = len(order_dict.keys())
        
    if static_rti_opts is not None:
        static_rti = RTI(static_rti_opts['look_back'], static_rti_opts['min_dist'], static_rti_opts['min_time'])
        cfg_dict['static_rti_cfg']   = simulation_utils.initializeBCI(base_opts)
        order_dict['static_rti_cfg'] = len(order_dict.keys())
        
    
    session_scores = np.zeros((base_opts['n_sessions'] + 1, len(cfg_dict.keys())))
        
    # Day 0 performance:
    D_dict = dict()
    for key, cfg in cfg_dict.items():
        D_key         = key.split('cfg')[0] + 'D'
        D_dict[D_key] = np.copy(cfg_dict[key]['D'])
        
    for i, (key, value) in enumerate(cfg_dict.items()):
        session_scores[0, order_dict[key]] = np.mean(simulateBCIFitts(value)['ttt'])

    # Now simulate full stretch of sessions
    for i in range(base_opts['n_sessions']):
        for j in range(base_opts['days_between'] + 1):
            
            # apply neural drift for each different method being tested: 
            for key, cfg in cfg_dict.items():
                cfg['neuralTuning'] = simulation_utils.simulateTuningShift(cfg['neuralTuning'], 
                                                                           n_stable = base_opts['n_stable'], 
                                                                           PD_shrinkage = base_opts['shrinkage'], 
                                                                           mean_shift = 0, 
                                                                           renormalize = simulation_utils.sampleSNR())  
        
        # No recalibration:
        D_dict['norecal_D'][:, 0] = D_dict['norecal_D'][:,0] / np.linalg.norm(D_dict['norecal_D'][1:, :][:, 0]) / np.linalg.norm(cfg_dict['norecal_cfg']['neuralTuning'][:, 1])
        D_dict['norecal_D'][:, 1] = D_dict['norecal_D'][:,1] / np.linalg.norm(D_dict['norecal_D'][1:, :][:, 1]) / np.linalg.norm(cfg_dict['norecal_cfg']['neuralTuning'][:, 2])        

        cfg_dict['norecal_cfg']['D']    = D_dict['norecal_D']
        cfg_dict['norecal_cfg']['beta'] = simulation_utils.gainSweep(cfg_dict['norecal_cfg'], possibleGain = base_opts['possibleGain'])
        
        idx                          = order_dict['norecal_cfg']
        session_scores[i+1, idx] = np.mean(simulateBCIFitts(cfg_dict['norecal_cfg'])['ttt'])
        
        # supervised: 
        cfg_dict['supervised_cfg']['D'] = simulation_utils.simulate_OpenLoopRecalibration(cfg_dict['supervised_cfg'], nSteps = 10000)
        cfg_dict['supervised_cfg']['D'] = simulation_utils.simulate_ClosedLoopRecalibration(cfg_dict['supervised_cfg'])
        idx                             = order_dict['supervised_cfg']
        
        cfg_dict['supervised_cfg']['beta'] = simulation_utils.gainSweep(cfg_dict['supervised_cfg'], 
                                                                       possibleGain = base_opts['possibleGain'])
        
        session_scores[i+1, idx]        = np.mean(simulateBCIFitts(cfg_dict['supervised_cfg'])['ttt'])
        
        # PRI-T recalibration:   
        if hmm_opts is not None:
            cfg_dict['hmm_cfg']['D']    = simulation_utils.simulate_HMMRecalibration(cfg_dict['hmm_cfg'], hmm)
            cfg_dict['hmm_cfg']['beta'] = simulation_utils.gainSweep(cfg_dict['hmm_cfg'], 
                                                                     possibleGain = base_opts['possibleGain'])
            idx                         = order_dict['hmm_cfg']
            session_scores[i+1, idx]    = np.mean(simulateBCIFitts(cfg_dict['hmm_cfg'])['ttt'])  
            
        # Static PRI-T recalibration:   
        if static_hmm_opts is not None:
            # Avoid storing the outcomes of recalibration - we're using the fixed decoder from day 0 
            # on every new day to start
            cfg_copy         = copy.deepcopy(cfg_dict['static_hmm_cfg'])
            cfg_copy['D']    = simulation_utils.simulate_HMMRecalibration(cfg_dict['static_hmm_cfg'], static_hmm)
            cfg_copy['beta'] = simulation_utils.gainSweep(cfg_dict['static_hmm_cfg'], possibleGain = base_opts['possibleGain'])
            
            idx                         = order_dict['static_hmm_cfg']
            session_scores[i+1, idx]    = np.mean(simulateBCIFitts(cfg_copy)['ttt'])  
            
        
        # FA stabilizer:
        if ss_opts is not None:
            cfg_dict['ss_cfg']['D'] = simulation_utils.simulate_LatentClosedLoopRecalibration(cfg_dict['ss_cfg'], ss_decoder_dict, 
                                                                                              stabilizer, ss_opts, daisy_chain = True)
            cfg_dict['ss_cfg']['beta'] = simulation_utils.gainSweep(cfg_dict['ss_cfg'], possibleGain = base_opts['possibleGain'])
            
            idx                      = order_dict['ss_cfg']
            session_scores[i+1, idx] = np.mean(simulateBCIFitts(cfg_dict['ss_cfg'])['ttt'])
            
        # Static FA stabilizer: 
        if static_ss_opts is not None:
            cfg_dict['static_ss_cfg']['D'] = simulation_utils.simulate_LatentClosedLoopRecalibration(cfg_dict['static_ss_cfg'], 
                                                                                                     static_ss_decoder_dict, 
                                                                                              static_stabilizer, static_ss_opts, daisy_chain = False)
            cfg_dict['static_ss_cfg']['beta'] = simulation_utils.gainSweep(cfg_dict['static_ss_cfg'], 
                                                                           possibleGain = base_opts['possibleGain'])
            
            idx                      = order_dict['static_ss_cfg']
            session_scores[i+1, idx] = np.mean(simulateBCIFitts(cfg_dict['static_ss_cfg'])['ttt'])
            
    
        # RTI 
        if rti_opts is not None:
            cfg_dict['rti_cfg']['D']    = simulation_utils.simulate_RTIRecalibration(cfg_dict['rti_cfg'], rti)
            cfg_dict['rti_cfg']['beta'] = simulation_utils.gainSweep(cfg_dict['rti_cfg'], possibleGain = base_opts['possibleGain'])
            idx                         = order_dict['rti_cfg']
            session_scores[i+1, idx]    = np.mean(simulateBCIFitts(cfg_dict['rti_cfg'])['ttt']) 
            
        # static RTI
        if static_rti_opts is not None:
            # Avoid storing the outcomes of recalibration - we're using the fixed decoder from day 0 
            # on every new day to start
            cfg_copy         = copy.deepcopy(cfg_dict['static_rti_cfg'])
            cfg_copy['D']    = simulation_utils.simulate_RTIRecalibration(cfg_dict['static_rti_cfg'], static_rti)
            cfg_copy['beta'] = simulation_utils.gainSweep(cfg_dict['static_rti_cfg'], possibleGain = base_opts['possibleGain'])
            idx                         = order_dict['static_rti_cfg']
            session_scores[i+1, idx]    = np.mean(simulateBCIFitts(cfg_copy)['ttt']) 
            

    return session_scores


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
base_opts['possibleGain']   = np.linspace(0.1,2.5,10)
base_opts['center_means']   = False
base_opts['nTrainingSteps'] = 10000

base_opts['n_sessions']   = 60   # number of sessions to simulate 
base_opts['days_between'] = 0    # days between session days
base_opts['shrinkage']    = 0.91  # relative tuning in subspace per new day
base_opts['n_stable']     = 0


# stabilizer settings:
ss_opts                 = dict()
ss_opts['B']            = 100
ss_opts['thresh']       = 0.05
ss_opts['n_components'] = 2
ss_opts['model_type']   = 'PCA'

# static stabilizer settings: 
static_ss_opts                 = dict()
static_ss_opts['B']            = 190
static_ss_opts['thresh']       = 0.1
static_ss_opts['n_components'] = 6
static_ss_opts['model_type']   = 'PCA'


# PRI-T settings:
hmm_opts = dict()
hmm_opts['probThresh']  = 'probWeighted'
hmm_opts['gridSize']    = 20
hmm_opts['stayProb']    = 0.999
hmm_opts['adjustKappa'] = lambda x: 1 / (1 + np.exp(-1 * (x - 0.) * 32.))
hmm_opts['vmKappa']     = 2

# static PRI-T settings:
static_hmm_opts = dict()
static_hmm_opts['probThresh']  = 'probWeighted'
static_hmm_opts['gridSize']    = 20
static_hmm_opts['stayProb']    = 0.999
static_hmm_opts['adjustKappa'] = lambda x: 1 / (1 + np.exp(-1 * (x - 0.5) * 8.8))
static_hmm_opts['vmKappa']     = 1


# RTI settings:
rti_opts = dict()
rti_opts['look_back'] = 240
rti_opts['min_dist']  = 0
rti_opts['min_time']  = 30

# Static RTI settings:
static_rti_opts = dict()
static_rti_opts['look_back'] = 320
static_rti_opts['min_dist']  = 0.1
static_rti_opts['min_time']  = 50


# ============================================================


parser = argparse.ArgumentParser(description = 'Code for measuring performance across time.')
parser.add_argument('--n_jobs', type = int, help = 'Number of jobs running this script')
parser.add_argument('--jobID', type = int, help = 'job ID')
parser.add_argument('--saveDir', type = str, default = './', help = 'Folder for saving scores')
args  = parser.parse_args()


if __name__ == '__main__':
    
    session_scores          = np.zeros((reps, base_opts['n_sessions'] + 1, 6)) 

    #reps = 10
    #base_opts['n_sessions']   = 15   # number of sessions to simulate 
    
    hmm_opts = None
    ss_opts  = None
    rti_opts = None

    sweep_args = [[base_opts, hmm_opts, static_hmm_opts, ss_opts, static_ss_opts, rti_opts, static_rti_opts]] * reps
    job_args   = np.array_split(sweep_args, args.n_jobs)[args.jobID]
    save_fname = os.path.join(args.saveDir, f'sweep_scores_{args.jobID}.npy')
    
    print('Number of runs for this job: ', len(job_args))
    print('File save path:', save_fname)
    
    rep_data = Parallel(n_jobs= -1, verbose = 11)(delayed(simulate_MultiSessionStretch)(*x) for x in job_args)
    np.save(save_fname, rep_data)
    print('Finished.')




