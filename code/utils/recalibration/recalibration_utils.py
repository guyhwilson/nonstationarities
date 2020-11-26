import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sys
#sys.path.append('utils/MATLAB/')
sys.path.append('utils/preprocessing/')
from preprocess import DataStruct, daysBetween




def traintest_DecoderSupervised(train_x, test_x, train_y, test_y, decoder = None, meanRecal = False):
    '''
    Code for training and testing linear decoder on provided data. Inputs are:
        
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
        decoder    = LinearRegression(fit_intercept = False, normalize = False).fit(train_x_continuous - train_means, train_y_continuous)
        
    if meanRecal:
        test_scores = [decoder.score(x - test_means, y) for x, y  in zip(test_x, test_y)] 
    else:
        test_scores = [decoder.score(x - train_means, y) for x, y  in zip(test_x, test_y)] 

    return test_scores, decoder