import numpy as np
from simulation import simulateBCIFitts


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




def orthogonalizeAgainst(v2, v1):
    '''Orthogonalize v2 against vector v1, keeping this latter vector the same.'''
    
    u2  = v2 - (v2.dot(v1) * v1)
    u2 /= np.linalg.norm(u2)
    
    return u2 * np.linalg.norm(v2)




def simulateTuningShift(tuning, PD_shrinkage, PD_noisevar = 1, mean_shift = 0, renormalize = True):
    ''' Simulate tuning shift for units. Inputs are:
    
        tuning (2D np array) - n_units x 3 array of tuning data
        PD_shrinkage (float) - relative strength of tuning in original PD
                               subspace following shift 
        PD_noisevar (float)  - PD noise variance; default = 1; irrelevant
                               if renormalize = True
        mean_shift (float)   - strength of average mean change 
        
    ''' 
    
    newTuning             = np.copy(tuning)
    newPD_component       = np.random.normal(loc = 0, scale = PD_noisevar**0.5, size = (tuning.shape[0], 2))
    newTuning[:, 0]      += np.random.normal(loc = 0, scale = mean_shift,     size = tuning.shape[0])
    
    if renormalize:  # adjust so that encoding norm same as earlier 
        
        tuningNorm        = np.mean(np.linalg.norm(tuning[:, 1:], axis = 0))  # will rescale to this norm
        newTuning[:, 1:] /= tuningNorm                                        # adjust to be unit vector for now 
        
        newPD_component, _    = np.linalg.qr(newPD_component, 'reduced')
        newPD_component[:, 0] = orthogonalizeAgainst(newPD_component[:, 0], tuning[:, 1]) 
        newPD_component[:, 1] = orthogonalizeAgainst(newPD_component[:, 1], tuning[:, 2]) 
        newTuning[:, 1:]      = (newTuning[:,1:] * PD_shrinkage) + (newPD_component *np.sqrt(1 - PD_shrinkage**2))  # unit norm vectors
        
        newTuning[:, 1:]     *= tuningNorm  # now scale up/down to match original data 
        
    else:  # use simpler approach: new = shrinkage * old + noise 
        newTuning[:,1:] = (tuning[:,1:] * PD_shrinkage) + newPD_component 
    
    return newTuning



def generateTargetGrid(gridSize, x_bounds = [-0.5, 0.5], y_bounds = [-0.5, 0.5]):
    '''
    Generate target grid for simulator.
    '''
    
    X_loc,Y_loc = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], gridSize), np.linspace(y_bounds[0], y_bounds[1], gridSize))
    targLocs    = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T
    
    return targLocs

def generateTransitionMatrix(gridSize, stayProb):
    '''
    Generate transition probability matrix for simulator targets.
    '''
    nStates     = gridSize**2
    stateTrans  = np.eye(nStates)*stayProb # Define the state transition matrix

    for x in range(nStates):
        idx                = np.setdiff1d(np.arange(nStates), x)
        stateTrans[x, idx] = (1-stayProb)/(nStates-1)

    pStateStart = np.zeros((nStates,1)) + 1/nStates
    
    return stateTrans, pStateStart

'''
def getDistortionMatrices_parallel(tuning, D, alpha, beta, nDelaySteps, delT, nSimSteps):
    Code for parallelizing HMM sweeps. Inputs are:
    
        inflection, exp (floats) - parameters for adjusting kappa weighting
        vmKappa (float)          - base kappa value
        probThresh (float)       - subselect high probability time points; between 0 and 1 
        decoder (sklearn)        - sklearn LinearRegression() object 
        neural (2D array)        - time x channels of neural activity
        stateTrans (2D array)    - square transition matrix for markov states
        targLocs (2D array)      - k x 2 array of corresponding target positions for each state
        B_cursorPos (2D array)   - time x 2 array of cursor positions
        pStateStart (1D array)   - starting probabilities for each state
    


    posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt = simulateBCIFitts(tuning, D, alpha, beta, nDelaySteps, delT, nSimSteps)    
    
    PosErr             = targTraj - posTraj
    D_new              = np.linalg.lstsq(np.hstack([np.ones((neuralTraj.shape[0],1)), neuralTraj]), PosErr, rcond = -1)[0]
    decVec             = np.hstack([np.ones((neuralTraj.shape[0],1)), neuralTraj]).dot(D_new)

    TargDist     = np.linalg.norm(PosErr, axis = 1)
    TargDir      = PosErr / TargDist[:, np.newaxis]
    farIdx       = np.where(TargDist > 0.4)[0]
    projVec      = np.sum(np.multiply(decVec[farIdx, :], TargDir[farIdx, :]), axis = 1)
    D_new       /= np.mean(projVec) 
    
    ttt = simulateBCIFitts(tuning, D_new, alpha, beta, nDelaySteps, delT, nSimSteps)[-1]
   
    dec  = D_new[1:, :]
    enc  = tuning[:, 1:]
    dec /= np.linalg.norm(enc, axis = 0)**2
    
   # enc /= np.linalg.norm(enc, axis = 0)
    
    distort = dec.T.dot(enc)
    
    return distort, ttt '''