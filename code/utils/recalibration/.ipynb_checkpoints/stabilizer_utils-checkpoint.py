import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import FactorAnalysis, PCA
from scipy.linalg import orthogonal_procrustes


def get_FactorAnalysisMap(fa):
    '''Get transformation matrix that maps observeds into 
       mean latent estimates. Inputs are:
       
           fa (sklearn object) - trained factor analysis model
        
       Returns:
           transform (2D array) - observed x latents matrix; apply to mean-centered data
    '''
    
    Ih    = np.eye(len(fa.components_))
    Wpsi  = fa.components_ / fa.noise_variance_
    cov_z = np.linalg.inv(Ih + np.dot(Wpsi, fa.components_.T))
    
    transform = np.dot(Wpsi.T, cov_z)
    return transform



def get_FA_ExplainedVariance(fa):
    '''Calculate explained variance ratios for each component in a FactorAnalysis model and 
    store in a explained_variance_ratio_ field (like sklearn). Inputs are:

    fa (FactorAnalysis) - trained FA model from sklearn library  '''

    n_components = fa.components_.shape[0]
    m            = fa.components_**2
    n            = fa.noise_variance_
    m1           = np.sum(m, axis = 1)
    frac_var     = np.asarray([m1[i]/(np.sum(m1)+np.sum(n)) for i in range(n_components)])

    fa.explained_variance_ratio_ = frac_var
    return fa


def get_ConditionAveraged(datas, conditions):
    '''Given a data array of rasters for different conditions, perform
    trial-averaging for trials in the same condition type. Inputs are:

    datas (list)          - list of time x channels arrays
    conditions (1D array) - trial labels  '''
    
    datas                   = np.dstack(datas)
    assert len(conditions) == datas.shape[2], "<Conditions> trial labels is not same length as data.shape[2] (data should be time x channels x trials)."

    n_samples, n_channels, n_trls = datas.shape
    unique                        = np.unique(conditions)
    avgd_data                     = np.dstack([datas[:, :, np.where(conditions == c)[0]].mean(axis = 2) for c in unique])

    return avgd_data, unique


def fit_ConditionAveragedModel(model_type, model_params, datas, conditions):
    '''Fit PCA or Factor Analysis model using condition-averaged and concatenated 
    data. Inputs are:

    model_type (str)      - either 'PCA' or 'FactorAnalysis'
    model_params (dict)   - dictionary of model parameters to pass
    data (list)           - list of time x channels arrays 
    conditions (1D array) - trial labels    '''

    assert model_type in ['PCA','FactorAnalysis'], "Model type not recognized."

    # prepare condition-averaged data:
    avgd_data, unique     = get_ConditionAveraged(datas, conditions)
    n_samples, n_channels = avgd_data.shape[0], avgd_data.shape[1]
    concat_data           = avgd_data.transpose((2, 0, 1)).reshape(n_samples * len(unique), n_channels)

    # generate model:
    model = eval(model_type)(**model_params)
    model.fit(concat_data)

    return model, model.score(concat_data)



def fit_TrialConcatenatedModel(model_type, model_params, data, sigma = None):
    '''
    Fit PCA or Factor Analysis model using condition-averaged and concatenated 
    data. Inputs are:

    model_type (str)      - either 'PCA' or 'FactorAnalysis'
    model_params (dict)   - dictionary of model parameters to pass
    data (list)           - entries are channels x time arrays  
    '''
    assert model_type in ['PCA','FactorAnalysis'], "Model type not recognized."

    # prepare trial-concatenated data:
    if sigma is not None:
        data = [gaussian_filter1d(trldat, sigma, axis = 0) for trldat in data]
        
    input_data = np.vstack(data)

    # generate model:
    model = eval(model_type)(**model_params)
    model.fit(input_data)

    return model, model.score(input_data)

  

def latentSweep(model_type, data, sweep_dims, sigma = None, model_params = dict()):
    '''
    Do a hyperparameter sweep to identify best latent dimensionality, based 
    upon performance on holdout data. Inputs are:

    model_type (str)          - can be 'FactorAnalysis' or 'PCA'
    datas (list)              - entries are channels x time arrays 
    sweep_dims (list of ints) - dimensionalities to test
    sigma (float)             - SD of gaussian kernel for data smoothing; default no smoothing
    model_params (dictionary) - shared parameters for all models generated in sweep

    Returns:

    sweep_results (np array)  - list containing with entries containing scores for each 
                                value in sweep_dims

    TODO:
    random_state (int)        - if provided, sets the random state for reproducibility or paired comparisons
    fit PCA using max(sweep_dims) then just subselect eigenvector #s according to sweep_dims
    '''
  
    sweep_len                     = len(sweep_dims)
    sweep_results                 = np.zeros((sweep_len,)) 
    models                        = list()

    for i, state_dim in enumerate(sweep_dims):
        model_params['n_components'] = state_dim

        model, score     = fit_TrialConcatenatedModel(model_type, model_params, data, sigma = sigma)
        sweep_results[i] = score
        models.append(model)

    return sweep_results, models




def identifyGoodChannels(lambda_1, lambda_2, B, thresh):
    '''
    Greedy algorithm from Degenhart 2020 for identifying good channels. Inputs are:

    lambda_1 (2D array) - channels x factors loadings matrix for first dataset
    lambda_2 (2D array) - channels x factors loadings matrix for second dataset
    B (int)             - number of good channels desired
    '''

    channel_set  = np.arange(lambda_1.shape[0])

    below_thresh = np.where(np.logical_or(np.linalg.norm(lambda_1, axis = 1) < thresh, np.linalg.norm(lambda_1, axis = 1) < thresh))[0]
    channel_set  = np.setdiff1d(channel_set, below_thresh)

    while len(channel_set) > B:
        lambda_1prime = lambda_1[channel_set, :]
        lambda_2prime = lambda_2[channel_set, :]

        O, _        = orthogonal_procrustes(lambda_1prime, lambda_2prime)
        delta       = lambda_1prime.dot(O) - lambda_2prime
        worst       = np.argmax(np.linalg.norm(delta, axis = 1))

        channel_set = np.setdiff1d(channel_set, channel_set[worst])

    return channel_set
    


class Stabilizer(object):
    """
    Stabilizer object from Degenhart et al. 2020. Initialized with data
    from a reference session. Can then be fit to a new session to obtain 
    a transformation into the reference latent space. 
    """
    
    def __init__(self, model_type, n_components):
        '''Inputs are:

          model_type (str)      - 'PCA' or 'FactorAnalysis'
          n_components (int)    - latent space dimensionality
        '''
        self.model_type   = model_type
        self.n_components = n_components

    def fit_ref(self, datas, conditionAveraged = False, conditions = None):
        '''Fit dimensionality reduction model to reference day data. Inputs are:

        datas (list)             - entries are time x channels arrays of neural activity
        conditionAveraged (Bool) - if True, average within conditions before concatenating
        conditions (list)        - labels for each trial; used only if conditionAveraged == True
        '''

        if conditionAveraged:
            trlens                         = [dat.shape[0] for dat in datas]
            assert len(np.unique(trlens)) == 1, "Trials have different lengths - unable to trial-average."
            model, ll  = fit_ConditionAveragedModel(self.model_type, {'n_components' : self.n_components}, datas, conditions)

        else:
            model, ll = fit_TrialConcatenatedModel(self.model_type, {'n_components' : self.n_components}, datas)

        self.ref_model  = model
        self.ref_coefs  = model.components_.T
        
        return self


    def fit_new(self, datas, B, thresh, conditionAveraged = False, conditions = None):
        '''Fit dimensionality reduction model to new day data as well as latent space mapping 
           to original day's subspace. Inputs are:

          datas (list)             - entries are time x channels arrays of neural activity
          B (int)                  - number of good channels desired
          thresh (float)           - cutoff for discarding noisy channels
          conditionAveraged (Bool) - if True, average within conditions before concatenating
          conditions (list)        - labels for each trial; used only if conditionAveraged == True
        '''

        if conditionAveraged:
            trlens                         = [dat.shape[0] for dat in datas]
            assert len(np.unique(trlens)) == 1, "Trials have different lengths - unable to trial-average."
            model, ll = fit_ConditionAveragedModel(self.model_type, {'n_components' : self.n_components}, datas, conditions)

        else:
            model, ll = fit_TrialConcatenatedModel(self.model_type, {'n_components' : self.n_components}, datas)

        self.new_model  = model
        self.new_coefs  = model.components_.T
        self.good_chans = identifyGoodChannels(self.ref_coefs, self.new_coefs, B, thresh)
        self.R, _       = orthogonal_procrustes(self.new_coefs[self.good_chans, :], self.ref_coefs[self.good_chans, :])
        self.B          = B
        self.thresh     = thresh


    def transform(self, data): 
        '''Transform neural data into latent space of reference day. Inputs are:

          data (2D array) - time x channels
        '''

        newdata_latent = self.new_model.transform(data).dot(self.R)
        return newdata_latent
    
    

class StabilizedPredictor(object):
    """
    Latent space decoder with stabilizer object. 
    """
    
    def __init__(self, model_type, n_components, regressor):
        '''Inputs are:

          model_type (str)      - 'PCA' or 'FactorAnalysis'
          n_components (int)    - latent space dimensionality
          regressor (object)    - regression model to use; must follow Sklearn model standard
                                  (specifically, train() and score() methods)
        '''
        self.stabilizer = Stabilizer(model_type, n_components)
        self.regressor  = regressor
        

    def fit_ref(self, datas, conditionAveraged = False, conditions = None):
        '''Fit dimensionality reduction model to reference day data. Inputs are:

        datas (list)             - entries are time x channels arrays of neural activity
        conditionAveraged (Bool) - if True, average within conditions before concatenating
        conditions (list)        - labels for each trial; used only if conditionAveraged == True
        '''

        self.stabilizer.fit_ref(datas, conditionAveraged, conditions)
        return self


    def fit_new(self, datas, B, thresh, conditionAveraged = False, conditions = None):
        '''Fit dimensionality reduction model to new day data as well as latent space mapping 
           to original day's subspace. Inputs are:

          datas (list)             - entries are time x channels arrays of neural activity
          B (int)                  - number of good channels desired
          thresh (float)           - cutoff for discarding noisy channels
          conditionAveraged (Bool) - if True, average within conditions before concatenating
          conditions (list)        - labels for each trial; used only if conditionAveraged == True
        '''
        self.stabilizer.fit_new(datas, B, thresh, conditionAveraged, conditions)
        
        return self

    def train(self, datas, targets):
        '''Fit regression model to latent space data on new session. Inputs are:
            
            datas (list)   - entries are time x channels arrays of neural activity
            targets (list) - entries are time x targets arrays of desired outputs
        '''
        assert hasattr(self.stabilizer, "new_model"), "New session's dimreduce model not fitted yet! Call fit_new() first."
        
        
        datas_all   = np.vstack(datas)
        targets_all = np.vstack(targets)
        latents_all = self.stabilizer.new_model.transform(datas_all)
        
        self.regressor.fit(latents_all, targets_all)
        return self
    
    def predict(self, datas):
        '''Predict target outputs from neural data. Note that <datas> is now an array here. Inputs are:
        
            datas (2D array) - time x channels array of neural activity
        '''
        
        assert hasattr(self.stabilizer, "new_model"), "New session's dimreduce model not fitted yet! Call fit_new() first."
        
        latents = self.stabilizer.new_model.transform(datas)
        preds   = self.regressor.predict(datas)
        
        return preds
        
        
    def score(self, datas, targets):
        '''Score regressor on paired neural and target recordings using its preferred scoring metric. Inputs are:
            
            datas (2D array)   - time x channels arrays of neural activity
            targets (2D array) - time x targets arrays of desired outputs
        '''
        
        score = self.regressor.score(datas, targets)
        return score
        