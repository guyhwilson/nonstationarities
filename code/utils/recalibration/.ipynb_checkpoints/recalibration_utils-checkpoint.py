import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import sys
#sys.path.append('utils/MATLAB/')
sys.path.append('utils/preprocessing/')
from preprocess import DataStruct, daysBetween




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
        decoder    = LinearRegression(fit_intercept = False, normalize = False).fit(train_x_continuous - train_means, train_y_continuous)
        
    if meanRecal:
        test_scores = [decoder.score(x - test_means, y) for x, y  in zip(test_x, test_y)] 
    else:
        test_scores = [decoder.score(x - train_means, y) for x, y  in zip(test_x, test_y)] 

    return test_scores, decoder



def get_WindowedFeatures(features, window, padding = 0):
	'''Use sliding window to generate subsampled features for a given target (e.g. go from instantaneous neural --> target
	   map to 200 msec neural window --> target). Inputs are:
	   
		   features (2D array)   - samples x features array of predictors
		   window (int)          - amount of spacing on each side of a time point window
		   padding (float)       - value to fill padded entries with 
	   
	'''
	
	n_samples, n_features = features.shape

	win_features  = np.zeros((n_samples, (2 * window + 1) * n_features))
	padding       = np.ones((window, n_features)) * padding
	features      = np.vstack([padding, features, padding])
	
	for i in range(n_samples):
		start = i 
		stop  = i + (2 * window) + 1
		
		sample_x           = features[start:stop, :]
		win_features[i, :] = sample_x.flatten()
    
	return win_features 



class ClickDecoder(object):
	'''Logistic regression pipeline for click decoding. Steps are PCA --> feature binning --> logreg. '''
	def __init__(self, window, n_components):
		self.window      = window
		self.n_component = n_components
		self.PCA         = PCA(n_components)
		self.classifier  = LogisticRegression(class_weight = 'balanced', max_iter = 3000)
		self.trained     = False
	
	def process_features(self, features):		
		features    = self.PCA.transform(features)
		features    = get_WindowedFeatures(features, window = self.window, padding = 0)
		
		return features

	def train(self, features, targets):
		self.n_init_features = features.shape[1]
		
		# Train PCA model on average activity around click times:
		pca_window   = 20
		click_times  = np.where(targets == 1)[0]
		click_times  = click_times[np.logical_and(click_times > pca_window, click_times < features.shape[0] - pca_window)]
		neural_click = np.zeros(((pca_window * 2) + 1, self.n_init_features, len(click_times)))

		for i, t in enumerate(click_times):
			neural_click[:, :, i] = features[(t - pca_window):(t + pca_window + 1), :]
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
	
	def predict_probs(self, features):
		features    = self.process_features(features)

		return self.classifier.predict_proba(features)
		


	

	
	