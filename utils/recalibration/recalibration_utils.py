import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import sys
#sys.path.append('../utils/preprocessing/')
from utils.preprocessing.preprocess import DataStruct, daysBetween



def get_BlockwiseMeanSubtracted(train_x, test_x, concatenate = True):
    '''Perform within-block mean subtraction for neural features at train time,
       and use prior block means at test time. Inputs are:
       
           train_x (list of 2D arrays) - entries are time x channels arrays for each block
                                         (concatenated across trials)
           test_x  (list of 2D arrays) - same as above.
           concatenate (bool)          - if True, combine blocks (within splits) before returning
    
    NOTE: assumes train_x and test_x are contiguous e.g. block IDs are [0, 1, 2, ], [3, 4]
    '''
    assert isinstance(train_x, list), 'train_x must be of type list'
    assert isinstance(test_x, list), 'test_x must be of type list'
    
    subtract = [train_x[-1]] + test_x
    train_x  = [x - x.mean(axis = 0) for x in train_x]
    test_x   = [subtract[i] - subtract[i-1].mean(axis = 0) for i in range(1, len(subtract))]
    
    if concatenate:
        train_x = np.concatenate(train_x)
        test_x  = np.concatenate(test_x)
        
    return train_x, test_x


def subtractMeans(train_x, test_x, method, concatenate = True):
    '''Perform within-block mean subtraction for neural features at train time,
       and use prior block means at test time. Inputs are:
       
           train_x (list of 2D arrays) - entries are time x channels arrays for each block
                                         (concatenated across trials)
           test_x  (list of 2D arrays) - same as above.
           method (string)             - can be:
                                           'overall'   - use entire training mean for train/test
                                           'blockwise' - use blockwise mean (prior block for test data)
                                           'rolling'   - 12 second window of activity within blocks
                                           
           concatenate (bool)          - if True, combine blocks (within splits) before returning
    '''
    
    train_x = train_x.copy()
    test_x  = test_x.copy()
    
    assert isinstance(train_x, list), 'train_x must be of type list'
    assert isinstance(test_x, list), 'test_x must be of type list'
    
    if method == 'overall':
        overall = np.concatenate(train_x).mean(axis = 0)
        train_x = [x - overall for x in train_x]
        test_x  = [x - overall for x in test_x]
    
    elif method == 'blockwise':
        subtract = [train_x[-1]] + test_x
        train_x  = [x - x.mean(axis = 0) for x in train_x]
        test_x   = [subtract[i] - subtract[i-1].mean(axis = 0) for i in range(1, len(subtract))]
        
    elif method == 'rolling':
        train_x_new, test_x_new = list(), list()
        subtract = [train_x[-1]] + test_x
        
        for i, x in enumerate(train_x):
            pad_val      = x.mean(axis = 0)
            running_mean = firingrate.rolling_window(x, window_size = 600, padding = pad_val).mean(axis = 1) 
            x_rolling    = x - running_mean
            train_x_new.append(x_rolling)
            
        for i, x in enumerate(test_x):
            pad_val      = subtract[i].mean(axis = 0)
            running_mean = firingrate.rolling_window(x, window_size = 600, padding = pad_val).mean(axis = 1) 
            x_rolling    = x - running_mean
            test_x_new.append(x_rolling)
            
        train_x = train_x_new
        test_x  = test_x_new 
        
    else:
        raise ValueError
            
    if concatenate:
        train_x = np.concatenate(train_x)
        test_x  = np.concatenate(test_x)
        
    return train_x, test_x




def traintest_DecoderSupervised(train_x, test_x, train_y, test_y, decoder = None, meanRecal = False):
    '''Code for training and testing linear decoder on provided data. Inputs are:
        
        train_x, test_x (list of 2D arrays) - entries are channels x time arrays of neural data
        train_y, test_y (list of 2D arrays) - entries are 2 x time arrays of cursor data
        decoder (Sklearn object)            - decoder to use; if None then defaults to linear regression 
        meanRecal (Bool)                    - whether or not to adapt decoder to test set means 

    Returns: 
    
      test_scores (list) - entries are performance on test trials 
      decoder (Sklearn) - trained decoder 
    '''
    train_x_continuous = np.vstack(train_x)
    test_x_continuous  = np.vstack(test_x)
    train_y_continuous = np.vstack(train_y)
    
    train_means        = train_x_continuous.mean(axis = 0)
    test_means         = test_x_continuous.mean(axis = 0)
    
    if decoder is None:
        decoder    = LinearRegression(fit_intercept = True, normalize = False).fit(train_x_continuous - train_means, train_y_continuous)
        
    if meanRecal:
        test_scores = [decoder.score(x - test_means, y) for x, y  in zip(test_x, test_y)] 
    else:
        test_scores = [decoder.score(x - train_means, y) for x, y  in zip(test_x, test_y)] 

    return test_scores, decoder


