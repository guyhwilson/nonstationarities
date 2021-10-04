import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import sys
#sys.path.append('utils/MATLAB/')
sys.path.append('utils/preprocessing/')
from preprocess import DataStruct, daysBetween



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


