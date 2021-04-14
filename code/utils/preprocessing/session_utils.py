import numpy as np
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d

import sys
sys.path.append('utils/MATLAB/')
sys.path.append('utils/preprocessing/')
#from HMM_utils import *
from preprocess import DataStruct, daysBetween



def get_Sessions(files, min_nblocks = 0, getClick = False):
    '''files (list of str) - list containing paths to each session's data
       min_nblocks (int)   - minimum number of blocks for a session to be included
       getClick (bool)     - only return sessions with click blocks
    '''
    
    sessions = list()
    for f in files:
        dat   = DataStruct(f)
        if getClick:
            if dat.decClick_continuous.max() > 0 and len(dat.blockList) >= min_nblocks:
                sessions.append(f)
        else:
            if len(dat.blockList) >= min_nblocks:
                sessions.append(f)
            
    
    return sessions



def get_SessionPairs(data_files, max_ndays, manually_remove = []):
    '''Generate pairs of sessions for recalibration analyses with user-specified restrictions
       on session characteristics. Inputs are:
       
           data_files (list of str)      - list containing paths to each session's data
           max_ndays (int)               - maximum number of days that sessions can be apart
           manually_remove (list of str) - remove specified files; defaults to no removal
    '''
    
    # remove manually specified by sessions:
    sessions   = np.setdiff1d(data_files, manually_remove)
    n_blocks   = np.zeros(( len(sessions), ))
        
    # keep pairs of sessions that are at most max_ndays apart
    pairs   = list()
    for i, A in enumerate(sessions):
        for j, B in enumerate(sessions):
            if i < j:
                date_A = A.split('t5.')[1].split('.mat')[0]
                date_B = B.split('t5.')[1].split('.mat')[0]
                if daysBetween(date_A, date_B) <= max_ndays:
                    pairs.append([A, B])
    return pairs
  

def getPairTasks(dayA, dayB, task):
    '''Try to get same task type on separate days, and default to all task types 
    if requested task not available on a given day. Inputs are:

    dayA, dayB (DataStruct) - sessions to use 
    task (str)              - task to try to use; use all if None

    Returns:
    dayA_task, dayB_task (str) - task; defaults to None if not available
    isSuccessful (int)         - 1 if both days have task type; 0 else 
    '''

    if task is not None:
        dayA_Has = sum(dayA.trialType == task) != 0 
        dayB_Has = sum(dayB.trialType == task) != 0 

        isSuccessful = dayA_Has * dayB_Has
        dayA_task    = [task if dayA_Has else None][0]
        dayB_task    = [task if dayB_Has else None][0]

    else:
        dayA_task    = None
        dayB_task    = None
        isSuccessful = 1

    return dayA_task, dayB_task, isSuccessful
  

def get_StrongTransferPairs(pairs, min_R2, train_frac, block_constraints = None, task = None):
    ''' Subselect session-pairs for those where a mean-adapted linear decoder transfers
        with some required threshold performance for a new day. Inputs are:

        pairs (list of tuples) - each tuple contains reference and new days' data file paths (str)
        min_R2 (float)         - minimum required performance of reference decoder on new day
        train_frac (float)     - portion of training data to use; between 0 and 1

        Also requires "block_constraints" dictionary which encodes which blocks to use in the following
        key-value pairs:

          <filename> (str) : (list of ints; block to use)

        If a filename is not contained in the key set, then all blocks from that session are okay to use.
    '''
    
    thresh_sessions = list()
    scores          = list()

    for i, (A_file, B_file) in enumerate(pairs):
        dayA      = DataStruct(A_file)
        dayB      = DataStruct(B_file)

        if block_constraints is not None:
            dayA_blocks = [block_constraints[A_file] if A_file in block_constraints.keys() else None][0]
            dayB_blocks = [block_constraints[B_file] if B_file in block_constraints.keys() else None][0]  
        else:
            dayA_blocks, dayB_blocks = None, None

        Atrain_x, Atest_x, Atrain_y, Atest_y = getTrainTest(dayA, train_frac = train_frac, blocks = dayA_blocks, task = task, return_flattened = True)
        try:
            Btrain_x, Btest_x, Btrain_y, Btest_y = getTrainTest(dayB, train_frac = train_frac, blocks = dayB_blocks, task = task, return_flattened = True)
        except:
            print(B_file, dayB_blocks)
            break

        lm        = LinearRegression(fit_intercept = False, normalize = False).fit(Atrain_x - Atrain_x.mean(axis = 0), Atrain_y)
        score     =  lm.score(Btest_x - Btest_x.mean(axis = 0), Btest_y)

        if score > min_R2:
            thresh_sessions.append([A_file, B_file])
            scores.append(score)

    return np.asarray(thresh_sessions), np.asarray(scores)


def getNeuralAndCursor(struct, sigma = None, task = None, blocks = None):
    '''
    Code for getting training and test data. Inputs are:

      struct (DataStruct)      - session to train on
      sigma (float)            - variance of gaussian filter to smooth neural data; default no smoothing
      task (str)               - task type to train and test on; defaults to all data   
      blocks (list of int)     - blocks to pull data from; default to all

    Returns:

    neural (list)    - entries are time x channels arrays of neural data 
    cursorErr (list) - entries are time x 2 arrays of cursor position error data
    targPos (list)   - entries are tuples containing target locations for each trial 
    '''

    if task == None:
        valid      = np.arange(len(struct.trialType))
    else:
        valid      = np.where(struct.trialType == task)[0]

    if blocks is not None:
        block_trls = np.where(np.in1d(struct.blockNums, blocks))[0]
        valid      = np.intersect1d(valid, block_trls)

    if sigma is not None:
        #print('Here?')
        neural     = [gaussian_filter1d(struct.TX[i].astype('float'), sigma, axis = 0) for i in valid]
    else:
        neural     = [struct.TX[i].astype('float') for i in valid]

    cursorErr           = [struct.targetPos[i] - struct.cursorPos[i] for i in valid]
    #targPos             = [struct.cursorPos[i] for i in valid]
    targPos             = [struct.targetPos[i][0] for i in valid]
    assert len(neural) == len(cursorErr), "Mismatch between number of trials for neural and cursor feature"

    return neural, cursorErr, targPos
  
  

def getTrainTest(struct, train_frac = 0.5, sigma = None, task = None, blocks = None, shuffle = False, return_flattened = False):
    '''
    Code for getting training and test data. Inputs are:

      struct (DataStruct)      - session to train on
      train_frac (float)       - fraction of trials to use on training 
      sigma (float)            - variance of gaussian filter to smooth neural data; default no smoothing
      task (str)               - task type to train and test on; defaults to all data   
      blocks (list of int)     - blocks to pull data from; default to all
      shuffle (bool)           - whether or not to shuffle trials before splitting into train/test
      return flattened (bool)  - if True, concatenate returned lists into 2D arrays

    Returns:

    train_x, test_x (list of 2D arrays) - entries are time x channels arrays of neural data 
    train_y, test_y (list of 2D arrays) - entries are time x 2 arrays of cursor position error data

    if return_flattened, then the above are concatenated in the time/samples dimension.
    '''

    neural, cursorErr, _   = getNeuralAndCursor(struct, sigma = sigma, task = task, blocks = blocks)
    n_trls                 = len(neural)
    train_ind, test_ind    = train_test_split(np.arange(n_trls), train_size = train_frac, shuffle = shuffle)

    train_x, test_x     = [neural[i] for i in train_ind], [neural[i] for i in test_ind]
    train_y, test_y     = [cursorErr[i] for i in train_ind], [cursorErr[i] for i in test_ind]

    if return_flattened:
        train_x = np.vstack(train_x)
        test_x  = np.vstack(test_x)
        train_y = np.vstack(train_y)
        test_y  = np.vstack(test_y)

    return train_x, test_x, train_y, test_y