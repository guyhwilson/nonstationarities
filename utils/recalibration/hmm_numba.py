import numpy as np
#import
import numba 
from numba import jit






@jit(nopython=False)
def hmmviterbi_vonmises_parallel(rawDecodeVec, stateTransitions, targLocs, cursorPos, pStateStart, vmKappa, adjustKappa = None, verbose = False):
    '''Run viterbi algorithm to find most likely sequence of target states given the cursor position and decoder outputs. Inputs are:

        rawDecodeVec (2D array)     - time x 2 array containing decoder outputs at each timepoint
        stateTransitions (2D array) - transition probabilities; n_states x n_states
        targLocs (2D array)         - n_states x 2 array containing corresponding target locations for each state
        cursorPos (2D array)        - time x 2 array of cursor positions
        pStateStart (vector)        - starting probabilities for each state 
        vmKappa (float)             - precision parameter for Von Mises observation model
        adjustKappa (method)        - fxn for weighting kappa values; defaults to None
        
    NOTE:
        - we could parallelize a lot of the below code (in the for loop) but it doesn't seem to result in substantial speed gains!
            - around ~40 sec run vs 38 sec run during testing
            - would need to fully cythonize to see substantial compute time decreases
    '''
    
    if adjustKappa is None:
        vmKappa_adjusted = vmKappa
        def adjustKappa(dist):
            return np.ones(dist.shape)

    numStates    = len(stateTransitions)
    L            = rawDecodeVec.shape[0]
    currentState = np.zeros((L, ))
    pTR          = np.zeros((numStates, L))

    # work in log space to avoid numerical issues
    logTR = np.log(stateTransitions)
    v     = np.log(pStateStart)
    vOld  = v

    # Precompute some values for speedup: 
    observedAngle    = np.arctan2(rawDecodeVec[:, 1], rawDecodeVec[:, 0])
    tDists           = np.linalg.norm(targLocs - cursorPos[:, np.newaxis, :], axis = 2)
    normPosErr       = (targLocs[:, np.newaxis] - cursorPos) / tDists.T[:, :, np.newaxis]
    expectedAngle    = np.arctan2(normPosErr[:, :, 1], normPosErr[:, :, 0])
    
    vmKappa_adjusted = vmKappa * adjustKappa(tDists)
    vmProbLog        = (vmKappa_adjusted * np.cos(observedAngle - expectedAngle).T) - np.log(2 * np.pi * np.i0(vmKappa_adjusted))

    # loop through the model;  von mises emissions probabilities
    for count in range(L):
        tmpV          = vOld + logTR
        maxIdx        = np.argmax(tmpV, axis = 1)
        maxVal        = np.take_along_axis(tmpV, np.expand_dims(maxIdx, axis=-1), axis=-1).squeeze(axis=-1)
        
        pTR[:,count] = maxIdx
        v            = vmProbLog[count, :] + maxVal
        vOld         = v
    
    # decide which of the final states is most probable
    finalState = np.argmax(v)
    logP       = v[finalState]

    # Now back trace through the model
    currentState[L - 1] = finalState

    for count in reversed(range(0, L - 1)):
        currentState[count] = pTR[int(currentState[count + 1]), count + 1]
        if currentState[count] == 0 & verbose == True:
            print('stats:hmmviterbi:ZeroTransitionProbability', currentState[ count + 1 ])

    return currentState, logP
