import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor

#def bootstrapMean(datas, n_bootstraps, CI = 0.95)


def bootstrap_LinearRegression(x, y, regressor = LinearRegression(), n_bootstraps = 1000, random_state = None):
  '''Bootstrap linear regression estimate of relationship between sets of variables. Inputs are:
  
    x (2D array)               - samples x features array of predictions
    y (1D array)               - samples x 1 array of targets
    regressor (sklearn object) - regression method; defaults to LinearRegression()
    n_bootstraps (int)         - number of bootstrap iterations to run
  
  '''
  
  n_samples, n_features = x.shape
  bootstrap_coefs       = np.zeros((n_bootstraps, n_features))
  bootstrap_means       = np.zeros((n_bootstraps, n_features))
  mean_lm               = regressor.fit(x, y)
  
  for i in range(n_bootstraps):
    sample_index = np.random.choice(n_samples, n_samples, replace = True)
    x_sample     = x[sample_index, :]
    y_sample     = y[sample_index]
    lm           = regressor.fit(x_sample, y_sample)
    
    bootstrap_means[i, :] = lm.intercept_
    bootstrap_coefs[i, :] = lm.coef_
    
    
  return mean_lm, bootstrap_means, bootstrap_coefs


    

