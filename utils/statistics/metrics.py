import numpy as np
from sklearn.linear_model import LinearRegression


def calcSNR(decoder, neural, cursorErr):
    ''' Estimate SNR by modeling decoder outputs as noisy linear redaout of point-at-target vector. 
        Inputs are:
        
            decoder (object)     - must have predict() method in sklearn style
            neural (2D array)    - time x channels array 
            cursorErr (2D array) - time x 2 array of target-cursor offsets 
    '''
    
    dists     = np.linalg.norm(cursorErr, axis = 1)
    p_t       = cursorErr / dists[:, np.newaxis]  # unit vector pointing at target
    
    u_t       = decoder.predict(neural)  
    mult      = np.linalg.lstsq(p_t, u_t, rcond = None)[0][0][0]
    res       = u_t - (p_t * mult)
    SNR       = mult / np.std(res)
    
    return SNR 
    


def estimateSNR(neural, cursor, target, trialStart, minDist, cutStart):
    '''Estimate SNR from cursor control data. Inputs are:
    
        neural (2D array)        - time x channels array 
        cursor (2D array)        - time x 2 array of cursor positions
        target (2D array)        - time x 2 array of target positions
        trialStart (1D iterable) - indices of new trial starts
        minDist (float)          - minimum distance for timepoint to be considered
        cutStart (int)           - avoid timesteps after each trial start
    '''
    
    assert neural.shape[0] == cursor.shape[0] and cursor.shape[0] == target.shape[0], "Different timelength inputs! Ensure that <neural>, <cursor>, and <target> have same leading dimension size."
    assert minDist >= 0, "MinDist must be nonnegative"
    assert cutStart >=0, "cutStart must be nonnegative"
    
    n_trials = len(trialStart)
    usedIdx  = list()
    
    for i in range(n_trials - 1):  # avoid very last trial
        start = int(trialStart[i])
        stop  = int(trialStart[i+1] - 1)

        trlTraj          = cursor[start:stop, :] 
        trlTraj_centered = trlTraj - cursor[start, :]
        trlTarg          = target[start:stop, :]

        dists     = np.linalg.norm(trlTraj - target[start, :], axis = 1)
        too_close = np.where(dists < minDist)[0]
        unused    = np.concatenate([np.arange(cutStart), too_close])
        used      = np.setdiff1d(np.arange(len(dists)),unused) + start
        usedIdx.append(used)
        
    usedIdx    = np.concatenate(usedIdx)
    cursorErr  = target - cursor
    lm         = LinearRegression().fit(neural[usedIdx, :], cursorErr[usedIdx, :])
    SNR        = calcSNR(lm, neural[usedIdx, :], cursorErr[usedIdx, :])
    
    return SNR, usedIdx