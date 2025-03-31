import numpy as np
import sys, glob, copy, os
import argparse

#[sys.path.append(f) for f in glob.glob('../utils/*')]
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

# for a reproducible result
np.random.seed(1)


##############################
ss_sweep_opts  = dict()  # stabilizer sweep settings
hmm_sweep_opts = dict()  # PRI-T sweep settings
rti_sweep_opts = dict()  # rti sweep settings

ss_sweep_opts['method']       = 'stabilizer'
ss_sweep_opts['thresh']       = [0.01, 0.03, 0.05, 0.07, 0.1]
ss_sweep_opts['n_components'] = np.arange(2, 8)
ss_sweep_opts['B']            = np.arange(10, 191, 30)
ss_sweep_opts['chained']      = [False, True]

hmm_sweep_opts['method']     = 'PRI-T'
hmm_sweep_opts['inflection'] = np.linspace(0, 0.5, 6)
hmm_sweep_opts['exp']        = np.linspace(1, 40, 6)
hmm_sweep_opts['kappa']      = [0.5, 1, 2, 3, 4, 6]
hmm_sweep_opts['chained']    = [False, True]

hmm_sweep_opts['method']    = 'RTI'
rti_sweep_opts['look_back'] = [200, 240, 280, 320, 360, 400, 500]
rti_sweep_opts['min_dist']  = [0, 0.1, 0.2, 0.3]
rti_sweep_opts['min_time']  = [10, 20, 30, 40, 50, 60]
rti_sweep_opts['chained']   = [False, True]


nReps = 30
nDays = 30

##############################


base_opts  = dict()  # unchanging parameters

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
base_opts['fixed_SNR']      = False

#base_opts['pause_likelihood'] = 0.003 # set to 0 for standard simulator 
#base_opts['newtarg_on_pause'] = 1 # set to 0 for standard simulator

base_opts['model_type']    = 'PCA'

base_opts['gridSize']   = 20
base_opts['probThresh'] = 'probWeighted'
base_opts['stayProb']   = 0.999


parser = argparse.ArgumentParser(description = 'Code for optimizing HMM across session pairs.')
parser.add_argument('--n_jobs', type = int, help = 'Number of jobs running this script')
parser.add_argument('--jobID', type = int, help = 'job ID')
parser.add_argument('--saveDir', type = str, default = './', help = 'Folder for saving scores')
args  = parser.parse_args()

    
def testHMM(base_opts, hmm_opts, n_days, n_reps):
    
    scores_dict = copy.deepcopy(hmm_opts)
    scores = np.zeros((n_reps))
    
    adjustKappa             = lambda x: 1 / (1 + np.exp(-1 * (x - hmm_opts['inflection']) * hmm_opts['exp']))
    targLocs                = hmm_utils.generateTargetGrid(gridSize = base_opts['gridSize'], is_simulated=True)
    stateTrans, pStateStart = hmm_utils.generateTransitionMatrix(gridSize = base_opts['gridSize'], stayProb = base_opts['stayProb'])

    hmm = HMMRecalibration(stateTrans, targLocs, pStateStart, hmm_opts['kappa'], adjustKappa = adjustKappa)
    
    for i in range(n_reps):
        cfg   = simulation_utils.initializeBCI(base_opts)
        
        for j in range(n_days):
            cfg['neuralTuning'] = simulation_utils.simulateTuningShift(cfg['neuralTuning'], 
                                                                       n_stable = base_opts['n_stable'], 
                                                                       PD_shrinkage = base_opts['shrinkage'], 
                                                                       mean_shift = 0,
                                                                       renormalize = simulation_utils.sampleSNR())  
            
            if hmm_opts['chained'] or j == n_days-1:
                # if using chaining, we update the decoder on each new day. If we're not chaining then
                # we'll only run the update on the very last day
                cfg['D']    = simulation_utils.simulate_HMMRecalibration(cfg, hmm)
                cfg['beta'] = simulation_utils.gainSweep(cfg, possibleGain = base_opts['possibleGain'])
                

        scores[i] = simulation_utils.evalOnTestBlock(cfg)
        
    scores_dict['ttt'] = scores

    return scores_dict



def testStabilizer(base_opts, ss_opts, n_days, n_reps):
    
    scores_dict = copy.deepcopy(ss_opts)
    scores      = np.zeros((n_reps))
    
    for i in range(n_reps):
        cfg, ss_decoder_dict, stabilizer = simulation_utils.initializeBCI({**base_opts, **ss_opts})
        cfg['neuralTuning'][:, 0] = 0

        for j in range(n_days):
            cfg['neuralTuning'] = simulation_utils.simulateTuningShift(cfg['neuralTuning'], n_stable = base_opts['n_stable'], 
                                                                       PD_shrinkage = base_opts['shrinkage'], mean_shift = 0, 
                                                                       renormalize = simulation_utils.sampleSNR())  
            cfg['D']    = simulation_utils.simulate_LatentClosedLoopRecalibration(cfg, ss_decoder_dict, stabilizer, ss_opts)
            cfg['beta'] = simulation_utils.gainSweep(cfg, possibleGain = base_opts['possibleGain'])
            
        scores[i] = simulation_utils.evalOnTestBlock(cfg)
        
    scores_dict['ttt'] = scores

    return scores_dict


def testRTI(base_opts, rti_opts, n_days, n_reps):
    
    scores_dict = copy.deepcopy(rti_opts)
    scores      = np.zeros((n_reps))
    rti         = RTI(rti_opts['look_back'], rti_opts['min_dist'], rti_opts['min_time'])
    
    for i in range(n_reps):
        cfg = simulation_utils.initializeBCI(base_opts)

        for j in range(n_days):
            cfg['neuralTuning'] = simulation_utils.simulateTuningShift(cfg['neuralTuning'], 
                                  n_stable = base_opts['n_stable'], PD_shrinkage = base_opts['shrinkage'], 
                                  mean_shift = 0, renormalize = simulation_utils.sampleSNR())  
            
            if rti_opts['chained'] or j == n_days-1:
                # if using chaining, we update the decoder on each new day. If we're not chaining then
                # we'll only run the update on the very last day
                cfg['D']    = simulation_utils.simulate_RTIRecalibration(cfg, rti)
                cfg['beta'] = simulation_utils.gainSweep(cfg, possibleGain = base_opts['possibleGain'])
            
        scores[i] = simulation_utils.evalOnTestBlock(cfg)
        
    scores_dict['ttt'] = scores

    return scores_dict
    

def testMethod(base_opts, method_opts, n_days, n_reps):
    
    if method_opts['method'] == 'stabilizer':
        scores_dict = testStabilizer(base_opts, method_opts, n_days, n_reps)
    elif method_opts['method'] == 'RTI':
        scores_dict = testRTI(base_opts, method_opts, n_days, n_reps)
    elif method_opts['method'] == 'PRI-T':
        scores_dict = testHMM(base_opts, method_opts, n_days, n_reps)
    else:
        raise ValueError('Method not recognized')
    
    return scores_dict
        

#--------------------------------------------------
if __name__ == '__main__':

    ss_args    = sweep_utils.generateArgs(ss_sweep_opts, {})
    hmm_args   = sweep_utils.generateArgs(hmm_sweep_opts, {})
    rti_args   = sweep_utils.generateArgs(rti_sweep_opts, {})
    sweep_args = np.array_split(np.concatenate([ss_args, hmm_args, rti_args]), args.n_jobs)[args.jobID]

    print('Stabilizer: {} parameters to sweep'.format(len(ss_args)))
    print('HMM: {} parameters to sweep '.format(len(hmm_args)))
    print('RTI: {} parameters to sweep '.format(len(rti_args)))
    print('Number for this job: ', len(sweep_args))
    
    sweep_scores = Parallel(n_jobs= -1, verbose = 5)(delayed(testMethod)(base_opts, x, nDays, nReps) for x in sweep_args)
    np.save(os.path.join(args.saveDir,  'sweep_scores_{}.npy'.format(args.jobID)), sweep_scores)
    print('Finished.')




