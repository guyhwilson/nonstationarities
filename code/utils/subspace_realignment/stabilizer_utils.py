import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import FactorAnalysis, PCA




def get_ConditionAveraged(data, conditions):
  '''
  Given a data array of rasters for different conditions, perform
  trial-averaging for trials in the same condition type. Inputs are:
  
    data (3D array)       - time x channels x trials array
    conditions (1D array) - trial labels
  '''
  assert len(conditions) == data.shape[2], "Trial labels not same length as data.shape[2], which should index trials."
  
  n_samples, n_channels, n_trls = data.shape
  unique                        = np.unique(conditions)
  avgd_data                     = np.dstack([data[:, :, np.where(conditions == c)[0]].mean(axis = 2) for c in unique])
  
  return avgd_data, unique


def fit_ConditionAveragedModel(model_type, model_params, data, conditions):
  '''
  Fit PCA or Factor Analysis model using condition-averaged and concatenated 
  data. Inputs are:
  
    model_type (str)      - either 'PCA' or 'FactorAnalysis'
    model_params (dict)   - dictionary of model parameters to pass
    data (3D array)       - time x channels x trials array 
    conditions (1D array) - trial labels
  '''
  assert model_type in ['PCA','FactorAnalysis'], "Model type not recognized."
  
  # prepare condition-averaged data:
  avgd_data, unique     = get_ConditionAveraged(data, conditions)
  n_samples, n_channels = avgd_data.shape[0], avgd_data.shape[1]
  concat_data           = avgd_data.transpose((2, 0, 1)).reshape(n_samples * len(unique), n_channels)
  
  # generate model:
  model = eval(model_type)(**model_params)
  model.fit(concat_data)
  
  return model, model.score(concat_data)

  

def latentSweep(model_type, data, conditions, sweep_dims, model_params = dict()):
  '''
  Do a hyperparameter sweep to identify best latent dimensionality, based 
  upon performance on holdout data. Inputs are:
  
    model_type (str)          - can be 'FactorAnalysis' or 'PCA'
    datas (3d array)          - time x channels x trials array
    conditions (1d array)     - contains trial goal type (e.g. in radial 8)
    sweep_dims (list of ints) - dimensionalities to test
    model_params (dictionary) - shared parameters for all models generated in sweep
    
  Returns:
      
    sweep_results (np array)  - #dimensions                                        
  
  TODO:
    random_state (int)        - if provided, sets the random state for reproducibility or paired comparisons
    fit PCA using max(sweep_dims) then just subselect eigenvector #s according to sweep_dims
  '''
  
  n_samples, n_channels, n_trls = data.shape
  sweep_len                     = len(sweep_dims)
  sweep_results                 = np.zeros((sweep_len,)) 
  models                        = list()
  
  for i, state_dim in enumerate(sweep_dims):
    model_params['n_components'] = state_dim

    model, score     = fit_ConditionAveragedModel(model_type, model_params, data, conditions)
    sweep_results[i] = score
    models.append(model)
   
  return sweep_results, models
      




