import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import FactorAnalysis, PCA
from scipy.linalg import orthogonal_procrustes
import copy
import scipy.ndimage





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
        self.ref_model    = None
        self.ref_coefs    = None
        self.new_model    = None
        self.new_coefs    = None
        

    def fit_ref(self, datas):
        '''Fit dimensionality reduction model to reference day data. Inputs are:
        datas (list)             - entries are time x channels arrays of neural activity; assumed centered
        '''

        model, ll = fit_TrialConcatenatedModel(self.model_type, {'n_components' : self.n_components}, datas)

        self.ref_model  = model
        self.ref_coefs  = model.components_.T
        
        return self


    def fit_new(self, datas, B, thresh, daisy_chain = False):
        '''Fit dimensionality reduction model to new day data as well as latent space mapping 
           to original day's subspace. Inputs are:
          datas (list)             - entries are time x channels arrays of neural activity; assumed centered
          B (int)                  - number of good channels desired
          thresh (float)           - cutoff for discarding noisy channels
        '''

        model, ll = fit_TrialConcatenatedModel(self.model_type, {'n_components' : self.n_components}, datas)

        if daisy_chain and self.new_model is not None:
            self.ref_model = copy.copy(self.new_model)
            self.ref_coefs = copy.copy(self.new_coefs)
            self.new_model  = model
            self.new_coefs  = model.components_.T
            self.good_chans = identifyGoodChannels(self.ref_coefs, self.new_coefs, B, thresh)
            R, _            = orthogonal_procrustes(self.new_coefs[self.good_chans, :], self.ref_coefs[self.good_chans, :])
            self.R          = R.dot(self.R)

        else:
            self.new_model  = model
            self.new_coefs  = model.components_.T
            self.good_chans = identifyGoodChannels(self.ref_coefs, self.new_coefs, B, thresh)
            self.R, _       = orthogonal_procrustes(self.new_coefs[self.good_chans, :], self.ref_coefs[self.good_chans, :])
            
        self.B          = B
        self.thresh     = thresh
        
        return self


    def transform(self, data): 
        '''Transform neural data into latent space of reference day. Inputs are:
          data (2D array) - time x channels; assumed centered
        '''

        newdata_latent = self.new_model.transform(data).dot(self.R)
        return newdata_latent
    
    
    def getNeuralToLatentMap(self, model):
        '''Get transformation matrix that maps observeds into latent estimates. Inputs are:
               model (sklearn object)  - trained factor analysis/PCA model
           Returns:
               transform (2D array) - observed x latents matrix; apply to mean-centered data
        '''
    
        if self.model_type == 'FactorAnalysis':
            Ih    = np.eye(len(model.components_))
            Wpsi  = model.components_ / model.noise_variance_
            cov_z = np.linalg.inv(Ih + np.dot(Wpsi, model.components_.T))
            transform = np.dot(Wpsi.T, cov_z)

        elif self.model_type == 'PCA':
            transform = model.components_.T
            
        else:
            raise ValueError('Model type not recognized')

        return transform
    


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x



def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0, causal = False):
    """1-D Gaussian filter.
    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode_reflect)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    
    Note: modified version of scipy function:
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/ndimage/filters.py#L210-L261
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    
    if causal:
        midpoint = int((len(weights) - 1) / 2)
        weights[(midpoint + 1):] = 0
        weights                 *= 2 # renormalize
        
    return scipy.ndimage.correlate1d(input, weights, axis, output, mode, cval, 0)
    
