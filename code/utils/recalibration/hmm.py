import numpy as np
import scipy



def hmmviterbi_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, pStateStart, vmKappa, adjustKappa = None, verbose = False):
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
    observedAngle_all = np.arctan2(rawDecodeVec[:, 1], rawDecodeVec[:, 0])

    # loop through the model;  von mises emissions probabilities
    for count in range(L):
        # 1. compute distance from the cursor to each target, and expected angle for that target
        tDists        = np.linalg.norm(targLocs - cursorPos[count, :], axis = 1)
        normPosErr    = (targLocs - cursorPos[count, :]) / tDists[:, np.newaxis]
        expectedAngle = np.arctan2(normPosErr[:, 1], normPosErr[:,0])

        # 2. compute expected precision based on the base kappa and distance to
        # target (very close distances -> very large dispersion in expected angles)
        vmKappa_adjusted = vmKappa * adjustKappa(tDists)

        # 3. compute VM probability densities
        observedAngle = observedAngle_all[count]
        vmProbLog     = (vmKappa_adjusted * np.cos(observedAngle - expectedAngle)) - np.log(2*np.pi* scipy.special.i0(vmKappa_adjusted))

        tmpV          = vOld + logTR
        #print(vOld.shape, logTR.shape)
        #print(pTR)
        maxIdx        = np.argmax(tmpV, axis = 1)
        maxVal        = np.take_along_axis(tmpV, np.expand_dims(maxIdx, axis=-1), axis=-1).squeeze(axis=-1)

        pTR[:,count] = maxIdx
        v            = vmProbLog + maxVal
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






def hmmdecode_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, pStateStart, vmKappa, adjustKappa = None, verbose = False):
    '''Run viterbi algorithm to find marginal probabilities of hidden states at each timestep (given observed data). Inputs are:

        rawDecodeVec (2D array)     - time x 2 array containing decoder outputs at each timepoint
        stateTransitions (2D array) - transition probabilities; n_states x n_states
        targLocs (2D array)         - n_states x 2 array containing corresponding target locations for each state
        cursorPos (2D array)        - time x 2 array of cursor positions
        pStateStart (vector)        - starting probabilities for each state 
        vmKappa (float)             - precision parameter for Von Mises observation model
        adjustKappa (method)        - fxn for weighting kappa values; defaults to None
    '''
    if adjustKappa is None:
        def adjustKappa(dist):
            return np.ones(dist.shape)
    
    numStates = len(stateTransitions)
    L         = rawDecodeVec.shape[0] + 1  # add extra symbols to start to make algorithm cleaner at f0 and b0
 
    # introduce scaling factors for stability
    fs      = np.zeros((numStates,L))
    fs[:,0] = pStateStart.squeeze()
    s       = np.zeros((L,))
    s[0]    = 1

    # Precompute some values for speedup: 
    observedAngle_all = np.arctan2(rawDecodeVec[:, 1], rawDecodeVec[:, 0])
    tDists_all        = np.linalg.norm(targLocs - cursorPos[:, np.newaxis, :], axis = 2)

    for count in range(1, L):
        # 1. compute distance from the cursor to each target, and expected angle for that target
        #tDists        = np.linalg.norm(targLocs - cursorPos[count - 1, :], axis = 1)
        tDists        = tDists_all[count - 1, :]
        normPosErr    = (targLocs - cursorPos[count - 1, :]) / tDists[:, np.newaxis]
        expectedAngle = np.arctan2(normPosErr[:, 1], normPosErr[:,0])

        # 2. compute expected precision based on the base kappa and distance to
        # target (very close distances -> very large dispersion in expected angles)
        vmKappa_adjusted = vmKappa * adjustKappa(tDists)

        # 3. compute VM probability densities
        observedAngle = observedAngle_all[count - 1]
        vmProbLog     = np.exp((vmKappa_adjusted * np.cos(observedAngle - expectedAngle)) - np.log(2*np.pi* scipy.special.i0(vmKappa_adjusted)))

        fs[:,count]   = vmProbLog * (stateTransitions.T.dot(fs[:, count-1]))
        s[count]      =  sum(fs[:,count])
        fs[:,count]  /=  s[count]

    bs = np.ones((numStates,L))
    for count in reversed(range(0, L - 1)):
        # 1. compute distance from the cursor to each target, and expected angle for that target
        tDists        = np.linalg.norm(targLocs - cursorPos[count, :], axis = 1)
        normPosErr    = (targLocs - cursorPos[count, :]) / tDists[:, np.newaxis]
        expectedAngle = np.arctan2(normPosErr[:, 1], normPosErr[:,0])

        # 2. compute expected precision based on the base kappa and distance to
        # target (very close distances -> very large dispersion in expected angles)
        vmKappa_adjusted = vmKappa * adjustKappa(tDists)

        # 3. compute VM probability densities
        observedAngle = observedAngle_all[count]
        vmProbLog     = np.exp((vmKappa_adjusted * np.cos(observedAngle - expectedAngle)) - np.log(2*np.pi* scipy.special.i0(vmKappa_adjusted)))

        probWeightBS = bs[:,count + 1] * vmProbLog
        tmp          = stateTransitions.dot(probWeightBS)
        bs[:,count]  = tmp * (1/s[count+1])

    pSeq = sum(np.log(s))
    pStates = fs * bs

    # get rid of the column that we stuck in to deal with the f0 and b0 
    pStates = pStates[:, 1:]

    #return s, fs
    return pStates, pSeq



def click_hmmviterbi_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, clickSignal, getClickProbs, pStateStart, vmKappa, adjustKappa = None, verbose = False):
    '''Run viterbi algorithm to find most likely sequence of target states given the cursor position and decoder outputs. Inputs are:

        rawDecodeVec (2D array)     - time x 2 array containing decoder outputs at each timepoint
        stateTransitions (2D array) - transition probabilities; n_states x n_states
        targLocs (2D array)         - n_states x 2 array containing corresponding target locations for each state
        cursorPos (2D array)        - time x 2 array of cursor positions
        clickSignal (vector)        - indicator for whether or not click detected at each time point (time x 1)
        getClickProbs (method)      - a function f: target_distance --> prob(Click | target_distance); outputs must be 
                                      bounded in [0, 1]
        pStateStart (vector)        - starting probabilities for each state 
        vmKappa (float)             - precision parameter for Von Mises observation model
        adjustKappa (method)        - fxn for weighting VonMises kappa parameter; default to none
    '''
    if adjustKappa is None:
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
    observedAngle_all = np.arctan2(rawDecodeVec[:, 1], rawDecodeVec[:, 0])


    # loop through the model;  von mises emissions probabilities
    for count in range(L):
        # 1. compute distance from the cursor to each target, and expected angle for that target
        tDists        = np.linalg.norm(targLocs - cursorPos[count, :], axis = 1)
        normPosErr    = (targLocs - cursorPos[count, :]) / tDists[:, np.newaxis]
        expectedAngle = np.arctan2(normPosErr[:, 1], normPosErr[:,0])

        # 2. compute expected precision based on the base kappa and distance to
        # target (very close distances -> very large dispersion in expected angles)
        vmKappa_adjusted = vmKappa * adjustKappa(tDists)

        # 3. compute VM probability densities
        observedAngle = observedAngle_all[count]
        vmProbLog     = (vmKappa_adjusted * np.cos(observedAngle - expectedAngle)) - np.log(2*np.pi* scipy.special.i0(vmKappa_adjusted))

        # 4. compute click probability densities
        observedClick = clickSignal[count]
        probClick     = getClickProbs(tDists)
        clickProbLog  = np.log(observedClick * probClick + ((1 - observedClick) * (1 - probClick)))
        
        if np.isnan(clickProbLog).any():
            print(count, observedClick, probClick, tDists.max())
            break
        vmProbLog    += clickProbLog.squeeze()

        tmpV          = vOld + logTR
        maxIdx        = np.argmax(tmpV, axis = 1)
        maxVal        = np.take_along_axis(tmpV, np.expand_dims(maxIdx, axis=-1), axis=-1).squeeze(axis=-1)

        pTR[:,count] = maxIdx
        v            = vmProbLog + maxVal
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



def click_hmmdecode_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, clickSignal, getClickProbs, pStateStart, vmKappa, adjustKappa = None, verbose = False):
    '''Run viterbi algorithm to find marginal probabilities of hidden states at each timestep (given observed data). Inputs are:

        rawDecodeVec (2D array)     - time x 2 array containing decoder outputs at each timepoint
        stateTransitions (2D array) - transition probabilities; n_states x n_states
        targLocs (2D array)         - n_states x 2 array containing corresponding target locations for each state
        cursorPos (2D array)        - time x 2 array of cursor positions
        getClickProbs (method)      - a function f: target_distance --> prob(Click | target_distance); outputs must be 
                                      bounded in [0, 1]
        pStateStart (vector)        - starting probabilities for each state 
        vmKappa (float)             - precision parameter for Von Mises observation model
        adjustKappa (method)        - fxn for weighting VonMises kappa parameter; default to none
    '''
    
    if adjustKappa is None:
        def adjustKappa(dist):
            return np.ones(dist.shape)
    
    numStates = len(stateTransitions)
    L         = rawDecodeVec.shape[0] + 1  # add extra symbols to start to make algorithm cleaner at f0 and b0

    # introduce scaling factors for stability
    fs      = np.zeros((numStates,L))
    fs[:,0] = pStateStart.squeeze()
    s       = np.zeros((L,))
    s[0]    = 1

    # Precompute some values for speedup: 
    observedAngle_all = np.arctan2(rawDecodeVec[:, 1], rawDecodeVec[:, 0])

    for count in range(1, L):
        # 1. compute distance from the cursor to each target, and expected angle for that target
        tDists        = np.linalg.norm(targLocs - cursorPos[count - 1, :])
        normPosErr    = (targLocs - cursorPos[count - 1, :]) / tDists
        expectedAngle = np.arctan2(normPosErr[:, 1], normPosErr[:,0])

        # 2. compute expected precision based on the base kappa and distance to
        # target (very close distances -> very large dispersion in expected angles)
        vmKappa_adjusted = vmKappa * adjustKappa(tDists)

        # 3. compute VM probability densities
        observedAngle = observedAngle_all[count - 1]
        vmProbLog     = np.exp((vmKappa_adjusted * np.cos(observedAngle - expectedAngle)) - np.log(2*np.pi* scipy.special.i0(vmKappa_adjusted)))

        # 4. compute click probability densities
        observedClick = clickSignal[count - 1]
        probClick     = getClickProbs(tDists)
        clickProbLog  = np.log(observedClick * probClick + ((1 - observedClick) * (1 - probClick)))

        if np.isnan(clickProbLog).any():
            print(count - 1, observedClick, probClick, tDists.max())
            break
        vmProbLog    += clickProbLog.squeeze()

        fs[:,count]   = vmProbLog * (stateTransitions.T.dot(fs[:, count-1]))
        s[count]      =  sum(fs[:,count])
        fs[:,count]  /=  s[count]
    
    bs = np.ones((numStates,L))

    for count in reversed(range(0, L - 1)):
        # 1. compute distance from the cursor to each target, and expected angle for that target
        tDists        = np.linalg.norm(targLocs - cursorPos[count, :])
        normPosErr    = (targLocs - cursorPos[count, :]) / tDists
        expectedAngle = np.arctan2(normPosErr[:, 1], normPosErr[:,0])

        # 2. compute expected precision based on the base kappa and distance to
        # target (very close distances -> very large dispersion in expected angles)
        vmKappa_adjusted = vmKappa * adjustKappa(tDists) 

        # 3. compute VM probability densities
        observedAngle = observedAngle_all[count]
        vmProbLog     = np.exp((vmKappa_adjusted * np.cos(observedAngle - expectedAngle)) - np.log(2*np.pi* scipy.special.i0(vmKappa_adjusted)))

        # 4. compute click probability densities
        observedClick = clickSignal[count]
        probClick     = getClickProbs(tDists)
        clickProbLog  = np.log(observedClick * probClick + ((1 - observedClick) * (1 - probClick)))

        if np.isnan(clickProbLog).any():
            print(count, observedClick, probClick, tDists.max())
            break
        vmProbLog    += clickProbLog.squeeze()

        probWeightBS = bs[:,count + 1] * vmProbLog
        tmp          = stateTransitions.dot(probWeightBS)
        bs[:,count]  = tmp * (1/s[count+1])

    pSeq = sum(np.log(s))
    pStates = fs * bs

    # get rid of the column that we stuck in to deal with the f0 and b0 
    pStates = pStates[:, 1:]

    #return s, fs
    return pStates, pSeq

