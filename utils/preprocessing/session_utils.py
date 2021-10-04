import numpy as np
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import scipy

import sys
#sys.path.append('utils/MATLAB/')
sys.path.append('utils/preprocessing/')
sys.path.append('utils/recalibration/')
from preprocess import DataStruct, daysBetween
from recalibration_utils import get_BlockwiseMeanSubtracted
import firingrate



def get_Dataset(dtype):
    ''''''



def get_Sessions(files, min_nblocks = 0, getClick = False, manually_remove = []):
    '''files (list of str)           - list containing paths to each session's data
       min_nblocks (int)             - minimum number of blocks for a session to be included
       getClick (bool)               - only return sessions with click blocks
       manually_remove (list of str) - remove specified files; defaults to no removal
    '''
    
    sessions = list()
    files    = np.setdiff1d(files, manually_remove)
    
    for f in files:
        dat   = DataStruct(f)
        if getClick:
            if dat.decClick_continuous.max() > 0 and len(dat.blockList) >= min_nblocks:
                sessions.append(f)
        else:
            if len(dat.blockList) >= min_nblocks:
                sessions.append(f)
            
    return sessions



def get_SessionPairs(sessions, max_ndays, manually_remove = []):
    '''Generate pairs of sessions for recalibration analyses with user-specified restrictions
       on session characteristics. Inputs are:
       
           sessions (list of str) - list containing paths to each session's data
           max_ndays (int)        - maximum number of days that sessions can be apart
    '''
    
    # remove manually specified by sessions:
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
  

def get_StrongTransferPairs(pairs, min_R2, train_size, sigma = None, block_constraints = None, task = None):
    ''' Subselect session-pairs for those where a mean-adapted linear decoder transfers
        with some required threshold performance for a new day. Inputs are:

        pairs (list of tuples) - each tuple contains reference and new days' data file paths (str)
        min_R2 (float)         - minimum required performance of reference decoder on new day
        train_size (float)     - portion of training data to use; between 0 and 1
        sigma (float)          - variance of gaussian filter to smooth neural data; default no smoothing

        Also requires "block_constraints" dictionary which encodes which blocks to use in the following
        key-value pairs:

          <filename> (str) : (list of ints; block to use)

        If a filename is not contained in the key set, then all blocks from that session are okay to use.'''
    
    thresh_sessions = list()
    scores          = list()

    for i, (A_file, B_file) in enumerate(pairs):
        dayA      = DataStruct(A_file)
        dayB      = DataStruct(B_file)
        
        Adate     = 't5.' + dayA.date
        Bdate     = 't5.' + dayB.date

        if block_constraints is not None:
            dayA_blocks = [block_constraints[Adate] if Adate in block_constraints.keys() else None][0]
            dayB_blocks = [block_constraints[Bdate] if Bdate in block_constraints.keys() else None][0]  
        else:
            dayA_blocks, dayB_blocks = None, None
            
        #print(dayA.date)
        #print(dayA_blocks, dayB_blocks)
        Atrain_x, Atest_x, Atrain_y, Atest_y = getTrainTest(dayA, train_size = train_size, task = task, blocks = dayA_blocks, sigma = sigma, shuffle = False, returnFlattened = True)
        Atrain_x, Atest_x                    = get_BlockwiseMeanSubtracted(Atrain_x, Atest_x, concatenate = True)
        Atrain_y                             = np.concatenate(Atrain_y)
        Atest_y                              = np.concatenate(Atest_y)
        
        Btrain_x, Btest_x, Btrain_y, Btest_y = getTrainTest(dayB, train_size = train_size, task = task, blocks = dayB_blocks, sigma = sigma, shuffle = False, returnFlattened = True)
        Btrain_x, Btest_x                    = get_BlockwiseMeanSubtracted(Btrain_x, Btest_x, concatenate = True)
        Btrain_y                             = np.concatenate(Btrain_y)
        Btest_y                              = np.concatenate(Btest_y)
        
        lm        = LinearRegression(fit_intercept = False, normalize = False).fit(Atrain_x, Atrain_y)
        score     = lm.score(Btest_x, Btest_y)
       # score     = scipy.stats.pearsonr(lm.predict(Btest_x).flatten(), Btest_y.flatten())[0]

        if score > min_R2:
            thresh_sessions.append([A_file, B_file])
            scores.append(score)

    return np.asarray(thresh_sessions), np.asarray(scores)


def getNeuralCursorTarget(struct, sigma = None, causal_filter = True, task = None, blocks = None):
    '''
    Code for getting training and test data. Inputs are:

      struct (DataStruct)      - session to train on
      sigma (float)            - variance of gaussian filter to smooth neural data; default no smoothing
      causal_filter (Bool)     - whether or not to use half-gaussian causal filter (default: False)
      task (str)               - task type to train and test on; defaults to all data   
      blocks (list of int)     - blocks to pull data from; default to all

    Returns:

    neural (list of lists) - one entry per block; sub-entries are time x channels arrays of neural data per trial
    cursorPos (list)       - one entry per block; sub-entries are time x 2 arrays of cursor positions per trial
    targPos (list)         - one entry per block; sub-entries are tuples containing target locations for each trial 
    
    TODO: 
        - enable subselection of time for each trial (wrt trial start)
    '''

    if task == None:
        valid      = np.copy(struct.blockList)
    else:
        valid      = struct.blockList[np.where(struct.gameName == task)[0]]
        
    if blocks is not None:
        valid      = np.intersect1d(valid, blocks)
        
    assert len(valid) > 0, "No blocks selected!"
    neural, cursorPos, targPos = list(), list(), list()
    
    for block in valid:
        neural_block    = list()
        cursorPos_block = list()
        targPos_block   = list()
        
        trls            = np.where(struct.blockNums == block)[0]
        for trl in trls:
            if sigma is not None:
                neural_block.append(firingrate.gaussian_filter1d(struct.TX[trl].astype('float'), sigma, axis = 0, causal = causal_filter))
            else:
                neural_block.append(struct.TX[trl].astype('float'))
            
            cursorPos_block.append(struct.cursorPos[trl])
            targPos_block.append(struct.targetPos[trl][-1])
            
        neural.append(neural_block)
        cursorPos.append(cursorPos_block)
        targPos.append(targPos_block)

    assert len(neural) == len(cursorPos), "Mismatch between number of trials for neural and cursor feature"

    return neural, cursorPos, targPos
  
  

def getTrainTest(struct, train_size = 0.67, sigma = None, causal_filter = True, task = None, blocks = None, shuffle = False, returnFlattened = False,
                 returnCursor = False):
    '''
    Code for getting training and test data. Inputs are:

      struct (DataStruct)      - session to train on
      train_size (float)       - fraction of blocks to use on training 
      sigma (float)            - variance of gaussian filter to smooth neural data; default no smoothing
      task (str)               - task type to train and test on; defaults to all data   
      blocks (list of int)     - blocks to pull data from; default to all
      shuffle (bool)           - whether or not to shuffle blocks before splitting into train/test
      returnFlattened (bool)   - if True, concatenate returned lists into 2D arrays
      returnCursor (bool)      - if True, return cursor position data as well
      
    Returns:

    train_x, test_x (list of 2D arrays) - entries are time x channels arrays of neural data 
    train_y, test_y (list of 2D arrays) - entries are time x 2 arrays of cursor position error data
    train_c, test_c (list of 2D arrays) - entries are time x 2 arrays of cursor position data (optional)

    if returnFlattened = True, then the above are concatenated in the time/samples dimension.
    
    TODO:
        - acoid cursor data generation if <returnCursor> set to False
    '''

    neural, cursorPos, targPos = getNeuralCursorTarget(struct, sigma = sigma, causal_filter = causal_filter, task = task, blocks = blocks)
    n_blocks                   = len(neural)
    train_blocks, test_blocks  = train_test_split(np.arange(n_blocks), train_size = train_size, shuffle = shuffle)
    
    # generate cursor error signal for regressing FRs against:
    cursorErr = list()
    for i in range(n_blocks):
        cursorErr.append([targ - cur for targ, cur in zip(targPos[i], cursorPos[i])])
    
    train_x, test_x = list(), list()
    train_y, test_y = list(), list()
    train_c, test_c = list(), list()
    
    # sort training data into output lists:
    for i in train_blocks:
        if returnFlattened:
            train_x.append(np.vstack(neural[i]))
            train_y.append(np.vstack(cursorErr[i]))
            train_c.append(np.vstack(cursorPos[i]))
        else:
            train_x.append(neural[i])
            train_y.append(cursorErr[i])
            train_c.append(cursorPos[i])
            
    # same code - maybe not as elegant but more readable for me at least
    for i in test_blocks:
        if returnFlattened:
            test_x.append(np.vstack(neural[i]))
            test_y.append(np.vstack(cursorErr[i]))
            test_c.append(np.vstack(cursorPos[i]))
        else:
            test_x.append(neural[i])
            test_y.append(cursorErr[i])
            test_c.append(cursorPos[i])
        
    if returnCursor:
        return train_x, test_x, train_y, test_y, train_c, test_c
    else:
        return train_x, test_x, train_y, test_y