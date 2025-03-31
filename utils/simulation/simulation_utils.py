import numpy as np
from sklearn.linear_model import LinearRegression
import copy 

import sys
sys.path.append('../utils/recalibration/')
from utils.simulation.simulation import simulateBCIFitts
from utils.recalibration import stabilizer_utils, RTI_utils



def generateUnits(n_units, SNR = 1):
    '''Generate PDs and baseline FRs for requested number of units.
    Inputs are: 
    
        n_units (int) - number of units to simulate
        SNR (float)   - norm of population tuning vector 
    
    Returns:
        
        tuning (2D array) - n_units x 3 array; 1st column has means 
                            while last two are tuning for x and y vel.
         
    We make sure that the last two columns are orthogonal (uniformly distributed PDs).
    '''
    
    tuning         = np.random.normal(size = (n_units, 3))
    tuning[:, 1:]  = np.linalg.qr(tuning[:, 1:], 'reduced')[0] * SNR
    
    return tuning


def simulateUnitActivity(tuning, noise, nSteps):
    '''Generate neural activity for random velocity movements. Inputs are:
        
        tuning (2D array) - channels x 3 array; 1st column = baselines, latter 
                            are x and y-velocity tuning coefficients
        noise (float)     - noise variance
        nSteps (int)      - number of timesteps to simulate'''
    
    nUnits         = tuning.shape[0]
    calVelocity    = np.random.normal(size = (nSteps, 2))
    calNeural      = calVelocity.dot(tuning[:,1:].T)  + np.random.normal(loc = tuning[:, 0].T, scale = noise, size = (nSteps, nUnits))  # FR = <velocity, PD> + baseline + noise 
    
    return calNeural, calVelocity


def sampleSNR(min_SNR = None):
    '''Draw from GMM mixture model fit to T5's SNR distribution.'''

    means = np.array([[2.78165778], [1.6244981 ]])
    covs  = np.array([[[0.07465756]], [[0.11382677]]])
    
    z = np.random.binomial(1, 0.6716436)
    y = np.random.normal(means[z], covs[z]**0.5)
    
    y *= 0.3365
    y -= 0.04438
    
    if min_SNR is not None:
        y = max(min_SNR, y)
    return y


def orthogonalizeAgainst(v2, v1):
    '''Orthogonalize v2 against vector v1, keeping this latter vector the same.'''
    
    u2  = v2 - (v2.dot(v1) * v1)
    u2 /= np.linalg.norm(u2)
    
    return u2 * np.linalg.norm(v2)




def simulateTuningShift(tuning, PD_shrinkage, PD_noisevar = 1, mean_shift = 0, 
                        n_stable = 0, renormalize = None):
    ''' Simulate tuning shift for units. Inputs are:
    
        tuning (2D np array) - n_units x 3 array of tuning data
        PD_shrinkage (float) - relative strength of tuning in original PD
                               subspace following shift 
        PD_noisevar (float)  - PD noise variance; default = 1; irrelevant
                               if renormalize = True
        mean_shift (float)   - strength of average mean change 
        
    ''' 
    n_units   = tuning.shape[0]
    newTuning = np.copy(tuning)
    oldnorm   = np.linalg.norm(tuning[:, 1:], axis = 0)
    
    # scale PD vectors to be unit norm 
    if renormalize is None:  
        renormalize = oldnorm
        
    newTuning[:, 1:]/= oldnorm     
    
    # generate perturbation vectors for means and PDs
    newPD_component   = np.random.normal(loc = 0, scale = PD_noisevar**0.5, size = (n_units, 2))
    newTuning[:, 0]  += np.random.normal(loc = 0, scale = mean_shift,       size = n_units)
   
    newPD_component, _    = np.linalg.qr(newPD_component, 'reduced')
    newPD_component[:, 0] = orthogonalizeAgainst(newPD_component[:, 0], tuning[:, 1]) 
    newPD_component[:, 1] = orthogonalizeAgainst(newPD_component[:, 1], tuning[:, 2]) 
    
    # combine (1 - alpha^2) of random orthogonal vector to get alpha correlation between old/new tuning subspaces 
    newPD_component[:n_stable, :] = newTuning[:n_stable, 1:]
    newTuning[:, 1:]              = (newTuning[:, 1:] * PD_shrinkage) + (newPD_component *np.sqrt(1 - PD_shrinkage**2)) 
    
    # rescale subspace norm 
    newTuning[:, 1:] /= np.linalg.norm(newTuning[:, 1:], axis=0)
    newTuning[:, 1:] *= renormalize 
    
    return newTuning



def gainSweep(cfg, possibleGain, verbose = False):
    '''Sweep through gain values to use.'''
    
    sweep_cfg                     = copy.deepcopy(cfg)
    sweep_cfg['pause_likelihood'] = 0 # set pauses to zero for gain optimization
    meanTTT                       = np.zeros((len(possibleGain),))

    for g in range(len(possibleGain)):
        sweep_cfg['beta'] = possibleGain[g]
        meanTTT[g]        = evalOnTestBlock(sweep_cfg)
        if verbose:
            print(str(g) + ' / ' + str(len(possibleGain)), 'gain = {:.1f}, ttt = {:.1f}'.format(possibleGain[g], meanTTT[g]))
        
    minIdx = np.argmin(meanTTT)
       
    return possibleGain[minIdx]



def renormalizeDecoder(D_new, cfg):
    
    if 'D' in cfg.keys():
        D_ref = np.linalg.norm(D_new[1:, :], axis = 0)[None, :] / np.linalg.norm(cfg['D'][1:, :], axis = 0)[None, :] 
        
    else:
        D_ref = np.linalg.norm(D_new[1:, :], axis = 0)[None, :]
    
    D_new /= D_ref
       
    return D_new


def evalOnTestBlock(cfg, test_n_timestamps = 20000):
    '''Evaluate trial times with fixed amount of data. Useful for when 
    sweeping nSimSteps but want same data for holdout eval. Inputs are:
    
        cfg (dict)              - configuration dictionary for simulator
        test_n_timestamps (int) - length in timesteps for eval block'''
    
    cfg_copy              = copy.deepcopy(cfg)
    cfg_copy['nSimSteps'] = test_n_timestamps
    cfg_copy['pause_likelihood'] = 0.
    
    ttt = np.mean(simulateBCIFitts(cfg_copy)['ttt']) 
    
    return ttt



def simulate_OpenLoopRecalibration(cfg, nSteps = 10000):
    
    neural_OL, posErr_OL = simulateUnitActivity(cfg['neuralTuning'], noise = 0.3, nSteps= nSteps)
    lr                   = LinearRegression(fit_intercept = True).fit(neural_OL, posErr_OL)
    D_OL                 = np.hstack([lr.intercept_[:, np.newaxis], lr.coef_ ]).T        
    
    D_OL = renormalizeDecoder(D_OL, cfg)
    
    return D_OL


def simulate_ClosedLoopRecalibration(cfg):
    
    calib_block  = simulateBCIFitts(cfg) 
    posErr       = calib_block['targTraj'] - calib_block['posTraj']
    neural       = calib_block['neuralTraj']
    
    lr           = LinearRegression(fit_intercept = True).fit(neural, posErr)
    D_CL         = np.hstack([lr.intercept_[:, np.newaxis], lr.coef_ ]).T        
    decVec_CL    = np.hstack([np.ones((neural.shape[0], 1)), neural]).dot(D_CL)
    
    D_CL = renormalizeDecoder(D_CL, cfg)

    return D_CL

def simulate_HMMRecalibration(cfg, hmm):
    
    calib_block = simulateBCIFitts(cfg) 
    
    if hmm.getClickProb is not None:        
        clickTraj = np.zeros((cfg['nSimSteps']))
        clickTraj[calib_block['trialStart']] =  1
        targStates, pTargState  = hmm.viterbi_search(calib_block['rawDecTraj'], calib_block['posTraj'], clickTraj)
    else:
        targStates, pTargState  = hmm.viterbi_search(calib_block['rawDecTraj'], calib_block['posTraj'])

    inferredTargLoc  = hmm.targLocs[targStates.astype('int').flatten(),:]
    inferredPosErr   = inferredTargLoc - calib_block['posTraj']
    neural           = calib_block['neuralTraj']

    lr             = LinearRegression(fit_intercept = True).fit(neural, inferredPosErr)
    D_HMM          = np.hstack([lr.intercept_[:, np.newaxis], lr.coef_ ]).T        
    D_HMM          = renormalizeDecoder(D_HMM, cfg)

    return D_HMM


def simulate_RTIRecalibration(cfg, rti):
    
    calib_block = simulateBCIFitts(cfg) 

    clickTraj = np.zeros((cfg['nSimSteps']))
    clickTraj[calib_block['trialStart']] =  1
    
    neural, inferredPosErr = rti.label(calib_block['neuralTraj'], calib_block['posTraj'], clickTraj)
    
    if neural.size != 0:
        lr             = LinearRegression(fit_intercept = True).fit(neural, inferredPosErr)
        D_RTI          = np.hstack([lr.intercept_[:, np.newaxis], lr.coef_ ]).T        
        D_RTI          = renormalizeDecoder(D_RTI, cfg)
        
    else:
        D_RTI = cfg['D']

    return D_RTI



def initializeBCI(base_opts):
    '''Initialize BCI simulator parameters. Input <base_opts> is a dictionary with items:
    
        'center_means' (str) : Bool - whether or not to include baseline firing rate term 
        'alpha' (str)        : float - exponential smoothing 
        'delT' (str)         : float - timestep size in seconds 
        'nDelaySteps' (str)  :
        'nSimSteps' (str)    :
        'nUnits'
        'SNR' 
        'n_components' (str)   : float - if not None, set up a latent space decoder 
        
        '''
    
    if 'SNR' not in base_opts.keys() or base_opts['SNR'] is None:
        SNR = sampleSNR()
    else:
        SNR = base_opts['SNR']
    
 
    initialTuning = generateUnits(n_units = base_opts['nUnits'], SNR = SNR)
    
    cfg = dict()
    cfg['alpha']        = base_opts['alpha'] 
    cfg['delT']         = base_opts['delT'] 
    cfg['nDelaySteps']  = base_opts['nDelaySteps']   
    cfg['nSimSteps']    = base_opts['nSimSteps']
    cfg['neuralTuning'] = initialTuning
    
    if base_opts['center_means']:
        cfg['neuralTuning'][:, 0] = 0
    
    if 'n_components' in base_opts.keys():
        assert 'model_type' in base_opts.keys() and 'n_components' in base_opts.keys(), "Missing some input parameters." 
        initialTuning[:, 0] = 0 # assume features are centered   
        calNeural, calVelocity    = simulateUnitActivity(cfg['neuralTuning'], noise = 0.3, nSteps = base_opts['nTrainingSteps'])
        decoder_dict, stabilizer  = fit_LatentDecoder(calNeural, calVelocity, base_opts)
        cfg['D']                  = decoder_dict['D']
        
    else:   
        cfg['D'] = simulate_OpenLoopRecalibration(cfg, base_opts['nTrainingSteps'] )
        
    cfg['beta'] = gainSweep(cfg, base_opts['possibleGain'], verbose = False)
         
    if 'n_components' in base_opts.keys():
        return cfg, decoder_dict, stabilizer
    
    else:
        return cfg



def fit_LatentDecoder(neural, posErr, args):
    '''Fit latent decoder.'''
    
    stab     = stabilizer_utils.Stabilizer(model_type = args['model_type'], n_components = args['n_components'])
    stab.fit_ref([neural])
    
    Q_ref    = stab.getNeuralToLatentMap(stab.ref_model)
    latents  = (neural - neural.mean(axis = 0)).dot(Q_ref)
    lr       = LinearRegression(fit_intercept = True).fit(latents, posErr)
     
    h        = lr.coef_                                               # map from latent space onto targets
    D_coef   = h.dot(Q_ref.T)                                         # compose to get neural --> latent --> output 
    D_latent = np.hstack([lr.intercept_[:, np.newaxis], D_coef]).T    # add bias terms
    
    D_latent = renormalizeDecoder(D_latent, cfg = dict())
    
    decoder_dict = dict()
    decoder_dict['Q'] = Q_ref    # neural --> latent
    decoder_dict['h'] = h        # latent --> velocity 
    decoder_dict['D'] = D_latent # neural --> latent --> velocity
    decoder_dict['lr_intercept'] = D_latent[0, :][:, None]

    return decoder_dict, stab


def recalibrate_LatentDecoder(neural, decoder_dict, stab, args):

    stab.fit_new([neural], B = args['B'], thresh = args['thresh'], daisy_chain = args['chained'])
    
    G_new        = stab.getNeuralToLatentMap(stab.new_model)
    D_coefnew    = decoder_dict['h'].dot(G_new.dot(stab.R).T)                # compose dimreduce with latent --> latent 
    D_new        = np.hstack([decoder_dict['lr_intercept'], D_coefnew]).T    # add bias terms
             
    return D_new
    
    
def simulate_LatentOpenLoopRecalibration(cfg, decoder_dict, stab, args):
    
    neural_OL, posErr_OL = simulateUnitActivity(cfg['neuralTuning'], noise = 0.3, nSteps = nSteps)
    
    D_latent = recalibrate_LatentDecoder(neural_OL, decoder_dict, stab, args)    
    D_latent = renormalizeDecoder(D_latent, cfg)

    return D_latent 
    
    
def simulate_LatentClosedLoopRecalibration(cfg, decoder_dict, stab, args, hmm = None):
    '''Unsupervised recalibration of latent space decoder.'''
    
    CL_block  = simulateBCIFitts(cfg)
    neural_CL = CL_block['neuralTraj']
    
    D_latent  = recalibrate_LatentDecoder(neural_CL, decoder_dict, stab, args)    
    D_latent  = renormalizeDecoder(D_latent, cfg)
    
    return D_latent



