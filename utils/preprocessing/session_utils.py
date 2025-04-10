import numpy as np
import glob, scipy, os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import date


import sys
#sys.path.append('utils/MATLAB/')
#sys.path.append('utils/preprocessing/')
#sys.path.append('utils/recalibration/')
from utils.preprocessing import firingrate
from utils.preprocessing.preprocess import DataStruct, daysBetween
from utils.recalibration.recalibration_utils import subtractMeans


def getBlockConstraints(FILE_DIR):

    noheadstill_dict             = dict()
    
    # old
    noheadstill_dict['t5.2016.09.26'] = [7, 12, 19, 21, 24, 29]
    noheadstill_dict['t5.2016.09.28'] = [4, 6, 7, 8, 9, 10, 16, 18, 19, 22, 24, 25, 29, 35]
    noheadstill_dict['t5.2016.10.03'] = [10, 14, 15, 27, 29, 30, 31]
    noheadstill_dict['t5.2016.10.05'] = [8, 9, 11, 12, 13, 14, 16, 17, 18, 21, 22]
    noheadstill_dict['t5.2016.10.07'] = [2, 3, 4, 5, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18]
    noheadstill_dict['t5.2016.10.10'] = [5, 8, 10, 11, 12, 14, 15, 16, 17, 20, 21, 22, 24, 25, 26, 28, 29, 30]
    noheadstill_dict['t5.2016.10.12'] = [4, 10, 12, 13, 14, 17, 18, 19, 22, 23, 25, 26, 28, 29, 30, 31]
    noheadstill_dict['t5.2016.10.13'] = [2, 3, 5, 7, 8, 9, 13, 14, 15, 17, 18, 19, 21, 22, 24]
    noheadstill_dict['t5.2016.10.17'] = [2, 4]
    noheadstill_dict['t5.2016.10.19'] = [2, 4]
    noheadstill_dict['t5.2016.10.24'] = [3, 5, 17, 19, 26, 28, 34, 38, 41, 45]
    noheadstill_dict['t5.2016.10.26'] = [3, 5, 9, 13, 23, 28, 34]
    noheadstill_dict['t5.2016.10.31'] = [3, 5, 6, 7, 8, 9]
    noheadstill_dict['t5.2016.12.06'] = [3, 5]
    noheadstill_dict['t5.2016.12.08'] = [5, 7]
    noheadstill_dict['t5.2016.12.15'] = [4, 8, 11, 12, 13, 14, 17, 18, 19]
    noheadstill_dict['t5.2016.12.16'] = [6, 8]
    noheadstill_dict['t5.2016.12.19'] = [3, 5]
    noheadstill_dict['t5.2016.12.21'] = [3, 6, 12, 13]
    noheadstill_dict['t5.2017.01.04'] = [4, 7]
    noheadstill_dict['t5.2017.01.30'] = [27, 28, 29]
    noheadstill_dict['t5.2017.02.15'] = [19, 21, 22, 23, 24]
    noheadstill_dict['t5.2017.02.22'] = [5, 6, 7, 8, 9, 10]
    noheadstill_dict['t5.2017.03.30'] = [10, 11, 12, 14]
    noheadstill_dict['t5.2017.04.26'] = [15, 16, 17, 18]
    noheadstill_dict['t5.2017.05.24'] = [16, 17, 18, 19]
    noheadstill_dict['t5.2017.05.31'] = [3, 5]
    noheadstill_dict['t5.2017.07.07'] = [2, 3]
    noheadstill_dict['t5.2017.07.31'] = [18, 19, 20, 21]
    noheadstill_dict['t5.2017.08.04'] = [7, 8]
    noheadstill_dict['t5.2017.08.07'] = [6, 7, 8]
    noheadstill_dict['t5.2017.09.20'] = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    noheadstill_dict['t5.2017.12.27'] = [4, 6, 13, 14, 15]
    noheadstill_dict['t5.2018.01.08'] = [0, 1, 2]
    noheadstill_dict['t5.2018.01.17'] = [2, 3]
    noheadstill_dict['t5.2018.01.19'] = [4]
    noheadstill_dict['t5.2018.01.22'] = [7]
    noheadstill_dict['t5.2018.01.24'] = [23, 24]
    noheadstill_dict['t5.2018.04.09'] = [4, 5, 6, 7, 8]
    noheadstill_dict['t5.2018.04.16'] = []
    noheadstill_dict['t5.2018.04.18'] = []
    noheadstill_dict['t5.2018.04.23'] = []
    noheadstill_dict['t5.2018.04.25'] = []
    noheadstill_dict['t5.2018.05.14'] = []
    noheadstill_dict['t5.2018.05.16'] = []
    noheadstill_dict['t5.2018.06.06'] = []
    noheadstill_dict['t5.2018.06.11'] = []
    noheadstill_dict['t5.2018.06.13'] = [4, 5, 6, 7, 8, 9]
    noheadstill_dict['t5.2018.06.25'] = []
    noheadstill_dict['t5.2018.06.27'] = []
    noheadstill_dict['t5.2018.07.02'] = []
    noheadstill_dict['t5.2018.07.06'] = []
    noheadstill_dict['t5.2018.07.11'] = []
    noheadstill_dict['t5.2018.07.25'] = [5, 6, 7, 8, 9]
    noheadstill_dict['t5.2018.07.27'] = []
    noheadstill_dict['t5.2018.08.06'] = []
    noheadstill_dict['t5.2018.08.08'] = []
    noheadstill_dict['t5.2018.08.22'] = [24, 25, 26, 27, 28, 29]
    noheadstill_dict['t5.2018.09.17'] = [24, 25, 27, 28, 30, 31]
    noheadstill_dict['t5.2018.10.24'] = [2, 3, 4, 5, 6]
    noheadstill_dict['t5.2018.11.21'] = []
    noheadstill_dict['t5.2018.11.28'] = []
    noheadstill_dict['t5.2018.12.12'] = []
    noheadstill_dict['t5.2018.12.17'] = []
    noheadstill_dict['t5.2018.12.19'] = [13, 14, 15, 16, 17]
    noheadstill_dict['t5.2019.01.09'] = []
    noheadstill_dict['t5.2019.01.14'] = []
    noheadstill_dict['t5.2019.01.16'] = []
    noheadstill_dict['t5.2019.01.23'] = [16, 17, 18, 19, 21]
    noheadstill_dict['t5.2019.01.30'] = []
    noheadstill_dict['t5.2019.02.22'] = [22, 23, 24, 25, 26]
    noheadstill_dict['t5.2019.03.13'] = []
    noheadstill_dict['t5.2019.03.20'] = [27, 28, 29, 30, 32]
    noheadstill_dict['t5.2019.03.27'] = [11, 18, 20, 22, 25, 27, 29]
    noheadstill_dict['t5.2019.04.01'] = []
    noheadstill_dict['t5.2019.04.03'] = []
    noheadstill_dict['t5.2019.04.08'] = []
    noheadstill_dict['t5.2019.04.29'] = [26, 27, 28, 29, 30]
    noheadstill_dict['t5.2019.05.29'] = [11, 12, 13, 14, 15]
    noheadstill_dict['t5.2019.06.19'] = [26, 27, 28, 29, 30]
    noheadstill_dict['t5.2019.07.01'] = [24, 25, 26, 27, 28]
    noheadstill_dict['t5.2019.09.18'] = []  # array reference issue day
    noheadstill_dict['t5.2019.10.28'] = [2, 4, 5, 6, 7] 
    noheadstill_dict['t5.2019.11.27'] = [31, 33, 36]
    noheadstill_dict['t5.2019.12.09'] = [22, 23, 24, 25, 26]
    noheadstill_dict['t5.2020.01.13'] = [26, 27, 28, 30]
    noheadstill_dict['t5.2020.02.26'] = []
    
    block_constraints = dict()

    for key, value in noheadstill_dict.items():
        new_key = os.path.join(FILE_DIR,'historical',f'{key}.mat')
        block_constraints[new_key] = value

    # new
    noheadstill_dict_new                  = dict()
    noheadstill_dict_new['t5.2021.07.26'] = [2, 3, 4]
    noheadstill_dict_new['t5.2021.07.07'] = [1, 2, 3]
    noheadstill_dict_new['t5.2021.06.02'] = [2, 3, 4] 
    noheadstill_dict_new['t5.2021.06.23'] = [1, 3]
    noheadstill_dict_new['t5.2021.05.26'] = [1, 2, 3]
    noheadstill_dict_new['t5.2021.07.14'] = [1, 3, 4]
    noheadstill_dict_new['t5.2021.07.08'] = [2, 3, 4]
    noheadstill_dict_new['t5.2021.05.05'] = [3, 4, 5]
    noheadstill_dict_new['t5.2021.05.17'] = [2, 3, 4]
    noheadstill_dict_new['t5.2021.04.26'] = [5, 6, 7]
    noheadstill_dict_new['t5.2021.07.19'] = [1, 2, 3]
    noheadstill_dict_new['t5.2021.07.12'] = [4, 5, 6]
    noheadstill_dict_new['t5.2021.06.07'] = [15, 16, 17]
    noheadstill_dict_new['t5.2021.05.19'] = [1, 2]
    noheadstill_dict_new['t5.2021.06.04'] = [16, 17, 18]
    noheadstill_dict_new['t5.2021.04.28'] = [2, 4, 5]
    noheadstill_dict_new['t5.2021.06.28'] = [1, 2, 3]
    noheadstill_dict_new['t5.2021.05.24'] = [2, 3, 4]
    noheadstill_dict_new['t5.2021.05.03'] = [6]
    noheadstill_dict_new['t5.2021.06.30'] = [5, 7]

    for key, value in noheadstill_dict_new.items():
        new_key = os.path.join(FILE_DIR,'new',f'{key}.mat')
        block_constraints[new_key] = value
    
    return block_constraints



def get_Sessions(files, min_nblocks = 0, getClick = False, block_constraints = None, manually_remove = []):
    '''files (list of str)           - list containing paths to each session's data
       min_nblocks (int)             - minimum number of blocks for a session to be included
       getClick (bool)               - only return sessions with click blocks
       manually_remove (list of str) - remove specified files; defaults to no removal
    '''
    
    sessions = list()
    files    = np.setdiff1d(files, manually_remove)
    
    for f in files:
        dat          = DataStruct(f)
        valid_blocks = np.copy(dat.blockList)
        
        # apply any block subselection
        if block_constraints is not None and f in block_constraints.keys():
            valid_blocks = np.intersect1d(valid_blocks, block_constraints[f])
        
        # todo: subselect blocks that have click activity
        if getClick:
            if dat.decClick_continuous.max() > 0 and len(valid_blocks) >= min_nblocks:
                sessions.append(f)
        else:
            if len(valid_blocks) >= min_nblocks:
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
        dayA      = DataStruct(A_file, causal_filter = sigma)
        dayB      = DataStruct(B_file, causal_filter = sigma)
        
        Adate     = 't5.' + dayA.date
        Bdate     = 't5.' + dayB.date

        if block_constraints is not None:
            dayA_blocks = [block_constraints[Adate] if Adate in block_constraints.keys() else None][0]
            dayB_blocks = [block_constraints[Bdate] if Bdate in block_constraints.keys() else None][0]  
        else:
            dayA_blocks, dayB_blocks = None, None
            
        Atrain_x, Atest_x, Atrain_y, Atest_y = getTrainTest(dayA, train_size = train_size, task = task, blocks = dayA_blocks, shuffle = False, returnFlattened = True)
        Atrain_x, Atest_x                    = subtractMeans(Atrain_x, Atest_x, method = 'blockwise', concatenate = True)
        Atrain_y                             = np.concatenate(Atrain_y)
        Atest_y                              = np.concatenate(Atest_y)
        
        Btrain_x, Btest_x, Btrain_y, Btest_y = getTrainTest(dayB, train_size = train_size, task = task, blocks = dayB_blocks, shuffle = False, returnFlattened = True)
        Btrain_x, Btest_x                    = subtractMeans(Btrain_x, Btest_x, method = 'blockwise', concatenate = True)
        Btrain_y                             = np.concatenate(Btrain_y)
        Btest_y                              = np.concatenate(Btest_y)
        
        lm        = LinearRegression(fit_intercept = False, normalize = False).fit(Atrain_x, Atrain_y)
        score     = lm.score(Btest_x, Btest_y)
       # score     = scipy.stats.pearsonr(lm.predict(Btest_x).flatten(), Btest_y.flatten())[0]

        if score >= min_R2:
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



def getFields(struct, fields, task = None, blocks = None):
    '''
    Code for getting training and test data. Inputs are:

      struct (DataStruct)      - session to train on
      task (str)               - task type to train and test on; defaults to all data   
      blocks (list of int)     - blocks to pull data from; default to all

    Returns dictionary with key strings corresponding to entries of <fields>:
    
        fields[i] (str) : (list of lists) - entry [i][j] is block i, trial j 
    '''

    if task == None:
        valid      = np.copy(struct.blockList)
    else:
        valid      = struct.blockList[np.where(struct.gameName == task)[0]]
        
    if blocks is not None:
        valid      = np.intersect1d(valid, blocks)
        
    assert len(valid) > 0, "No blocks selected!"
    
    results_dict = dict()
    for field in fields:
        results_dict[field] = list()
    
    for block in valid:
        trls        = np.where(struct.blockNums == block)[0]
        
        blocks_dict = dict()
        for field in fields:
            blocks_dict[field + '_block'] = list()
        
        for trl in trls:
            for field in fields:
                attr     = getattr(struct, field)
                attr_trl = attr[trl]
                
                if field == 'TX':
                    attr_trl = attr_trl.astype(float)
                
                if len(attr_trl.shape) == 1:
                    attr_trl = attr_trl[:, None]
                
                blocks_dict[field + '_block'].append(attr_trl)
        
        for field in fields:
            results_dict[field].append(blocks_dict[field + '_block'])

    return results_dict


def getTrainTest(struct, fields, train_size = 0.67, task = None, blocks = None, shuffle = False, returnFlattened = False):
    '''
    Code for getting training and test data. Inputs are:

      struct (DataStruct)      - session to train on
      train_size (float)       - fraction of blocks to use on training 
      sigma (float)            - variance of gaussian filter to smooth neural data; default no smoothing
      task (str)               - task type to train and test on; defaults to all data   
      blocks (list of int)     - blocks to pull data from; default to all
      shuffle (bool)           - whether or not to shuffle blocks before splitting into train/test
      returnFlattened (bool)   - if True, concatenate returned lists into 2D arrays
      
    Returns:

    train_x, test_x (list of 2D arrays) - entries are time x channels arrays of neural data 
    train_y, test_y (list of 2D arrays) - entries are time x 2 arrays of cursor position error data
    train_c, test_c (list of 2D arrays) - entries are time x 2 arrays of cursor position data (optional)

    if returnFlattened = True, then the above are concatenated in the time/samples dimension.
    '''
    

    results   = getFields(struct, fields, task = task, blocks = blocks)

    n_blocks                   = len(results[list(results.keys())[0]])
    train_blocks, test_blocks  = train_test_split(np.arange(n_blocks), train_size = train_size, shuffle = shuffle)
    
    # generate cursor error signal for regressing FRs against:
    traintest_dict = dict()
    for field in fields:
        traintest_dict['train_' + field] = list()
        traintest_dict['test_' + field] = list()
    

    # sort training data into output lists:
    for i in train_blocks:
        for field in fields:
            if returnFlattened:
                traintest_dict['train_' + field].append(np.vstack(results[field][i]))
            else:
                traintest_dict['train_' + field].append(results[field][i])
            
    # same code - maybe not as elegant but more readable for me at least
    for i in test_blocks:
        for field in fields:
            if returnFlattened:
                traintest_dict['test_' + field].append(np.vstack(results[field][i]))
            else:
                traintest_dict['test_' + field].append(results[field][i])
        
    return traintest_dict
  
  

