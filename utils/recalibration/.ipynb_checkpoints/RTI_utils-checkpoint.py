import numpy as np



def get_RTIData(neural, cursor, IsClick, lookback, MinDist = 0, MinTime = 0, ReturnInds = False):
    '''Generate neural features and target error signals via retrospective target
       inference (RTI) approach. Inputs are:
       
           neural (2D array)   - time x channels array of neural recordings
           cursor (2D array)   - time x 2 array of cursor positions
           IsClick (1D array)  - binary-valued array of decoded clicks
           lookback (int)      - number of previous timebins before a click to use 
           MinDist (float)     - exclude bins closer than this threshold from being selected (arbitrary units)
           MinTime (int)       - exclude bins within this time window of click from being selected (bins)
           ReturnInds (bool)   - whether or not to return the selected timepoints (default: false)
           
       Returns:
       
           features (2D array)     - time x channels of neural features
           targets (2D array)      - time x 2 array of (inferred) cursor error vectors
           selectedIdxs (1D array) - indexes of timepoints used from the original neural, cursor inputs (optional)
    '''
    
    assert len(IsClick.shape)  == 1, "Expected 1D array for input <isClick>"
    assert lookback > 0 and isinstance(lookback, int), "Lookback must be positive integer"
    assert MinDist >= 0, "MinDist must be nonnegative"
    
    success_clicks    = np.where(IsClick == 1)[0]
    features, targets = list(), list()
    selectedIdxs      = list()
    
    for i in success_clicks:
        if i > lookback:     # avoid clicks where not enough previous data to pull
            targ    = cursor[i, :]  
            start   = i - lookback        # look back at previous timepoints 
            stop    = i - MinTime + 1     # include only up to some temporal window prior to click 


            neural_retro  = neural[start:stop, :] 
            curErr_retro  = targ - cursor[start:stop, :] 
            dist_retro    = np.linalg.norm(curErr_retro, axis = 1)


            farIdx        = dist_retro >= MinDist        # time points spatially distant from inferred target
            approachIdx   = np.gradient(dist_retro) < 0  # time points where cursor heading toward inferred target
            #approachIdx   = np.ones((stop - start, ))  

            selectIdx     = np.where(np.logical_and(farIdx, approachIdx))[0]

            features.append(neural_retro[selectIdx, :]), targets.append(curErr_retro[selectIdx, :])
            selectedIdxs.append(start + selectIdx)

    features     = np.vstack(features)
    targets      = np.vstack(targets)
    selectedIdxs = np.concatenate(selectedIdxs)
    
    if ReturnInds:
        return features, targets, selectedIdxs
    else:
        return features, targets

