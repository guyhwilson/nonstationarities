import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import sys
sys.path.append('utils/preprocessing/')
from session_utils import get_StrongTransferPairs, get_SessionPairs
from preprocess import DataStruct, daysBetween





def get_T5_ClickTrainTest(dat, train_frac):
    '''Load T5 click decoder training and testing data, while accounting for dataset idiosyncrasies (-__-).
       Inputs are:
           
           dat (DataStruct)   - session data to use
           train_frac (float) - portion of data to use for training (overriden for 1 day with weird block structure)
    '''
    
    # weird exception - this day has the click probabilities (?) 
    # also has non-click blocks appearing halfway so we'll just select the two contiguous click periods for train and test
    if dat.date == '2016.10.31':
        dat.decClick_continuous = np.round(dat.decClick_continuous)
        
        train_x, test_x = dat.TX_continuous[:25000, :],    dat.TX_continuous[38500:47500, :]
        train_y, test_y = dat.decClick_continuous[:25000], dat.decClick_continuous[38500:47500]

    else:
        
        # day with weird nonstationarity (??); chop out affected portion near end
        if dat.date == '2016.10.24':
            dat.TX_continuous       = dat.TX_continuous[:50000, :]
            dat.decClick_continuous = dat.decClick_continuous[:50000]
        
        train_x, test_x, train_y, test_y = train_test_split(dat.TX_continuous, dat.decClick_continuous, train_size = train_frac, shuffle = False)
    
    return train_x, test_x, train_y, test_y




def get_WindowedFeatures(features, window, padding = 0):
    '''Use sliding window to generate subsampled features for a given target (e.g. go from instantaneous neural --> target
       map to 200 msec neural window --> target). Inputs are:
       
           features (2D array)   - samples x features array of predictors
           window (int)          - number of previous timepoints to use in window
           padding (float)       - value to fill padded entries with 
       
    '''
    
    n_samples, n_features = features.shape

    win_features  = np.zeros((n_samples, (window + 1) * n_features))
    padding       = np.ones((window, n_features)) * padding
    features      = np.vstack([padding, features, padding])
    
    for i in range(window, n_samples):
        sample_x           = features[(i - window):(i + 1), :]
        win_features[i, :] = sample_x.flatten()
    
    return win_features 





class ClickDecoder(object):
    '''Logistic regression pipeline for click decoding. Steps are PCA --> feature binning --> logreg. '''
    def __init__(self, window, n_components):
        self.window      = window                                                 # window used for decoder
        self.n_component = n_components                                           # PCA subspace dimensionality
        self.PCA         = PCA(n_components)                                      # PCA object
        self.classifier  = LogisticRegression(class_weight = 'balanced', max_iter = 3000)
        self.trained     = False
    
    def process_features(self, features):        
        features    = self.PCA.transform(features)
        features    = get_WindowedFeatures(features, window = self.window, padding = 0)
        
        return features

    def train(self, features, targets):
        self.n_init_features = features.shape[1]
        
        # Train PCA model on average activity around click times:
        click_times  = np.where(targets == 1)[0]
        click_times  = click_times[np.logical_and(click_times > self.window, click_times < features.shape[0] - self.window)]
        neural_click = np.zeros( (self.window + 1, self.n_init_features, len(click_times)) )

        for i, t in enumerate(click_times):
            neural_click[:, :, i] = features[(t - self.window):(t + 1), :]
        avgd = np.mean(neural_click, axis = 2)
        
        self.PCA.fit(avgd)
        
        # train downstream logistic regression:
        features    = self.process_features(features)    
        self.classifier.fit(features, targets)
        self.trained = True

        return self

    def predict(self, features):
        features    = self.process_features(features)

        return self.classifier.predict(features)
    
    def predict_proba(self, features):
        features    = self.process_features(features)

        return self.classifier.predict_proba(features)
        
    def score_delta(self, features, targets):
        '''Since click has huge class imbalance among other issues, we use another performance metric:
            the average time difference between a predicted click and the nearest actual click
        '''
        assert self.trained == True, "Model not trained! Run train() first."
        
        pred_targets = self.predict(features)
        pred_times   = np.where(pred_targets == 1)[0]
        true_times   = np.where(targets == 1)[0]

        time_diffs   = np.zeros((len(pred_times)))

        for i, t in enumerate(pred_times):
            nearest       = np.argmin(np.abs(t - true_times)) 
            diff          = t - true_times[nearest]
            time_diffs[i] = diff 
        
        return time_diffs

    
