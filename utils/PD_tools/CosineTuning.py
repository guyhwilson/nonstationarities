import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d
import scipy.stats
#from sklearn.linear_model import Lin


def normalizeDecoderGain(lm, decVec, posErr, thresh):
    targDist = np.linalg.norm(posErr, axis = 1)
    targDir  = posErr / targDist[:, np.newaxis]

    farIdx  = np.where(targDist > thresh)[0]

    projVec   = np.sum(np.multiply(decVec[farIdx, :], targDir[farIdx, :]), axis = 1)
    lm.coef_      /= np.mean(projVec)
    lm.intercept_ /= np.mean(projVec)
    
    return lm

def fitLinearModel(y_true, y_pred, threshold):
    
    dists     = np.linalg.norm(y_true, axis = 1)
    dist_mask = dists > threshold
    
    bias = y_pred[dist_mask, :].mean(axis = 0)
    
    y = (y_pred - bias)[dist_mask, :].flatten()
    x = (y_true[dist_mask, :] / dists[dist_mask, None]).flatten() 
    
    scale = x.dot(y) / np.sum(x**2) # normal equation reduces to this
    scale = max(scale, 1e-9) # dont allow negative SNR
    
    return scale, bias


def estimateNormalizedBias(decoder, x, y, threshold = 300, empirical_correction = True):
    scale, bias    = fitLinearModel(y, decoder.predict(x), threshold)
    
    if empirical_correction:
        bias -= y.mean(axis = 0)
        
    bias_index     = np.linalg.norm(bias) / scale
    
    return bias_index


def estimateSubspaceDrift(ref_coefs, new_coefs):
    '''Inputs are:
        
        ref_coefs (2D array) - n_vars x nChannels array of encoding/decoding weights
        new_coefs (2D array) - same as above '''
    
    assert ref_coefs.shape == new_coefs.shape, 'inputs must be same size'
        
    angles = list()
    
    for i in range(ref_coefs.shape[0]):
        drift_angle = angle_between(ref_coefs[i, :], new_coefs[i, :])
        angles.append(drift_angle)
        
    overall_angle = angle_between(ref_coefs.flatten(), new_coefs.flatten())
        
    return angles, overall_angle


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2, returnFullAngle = False):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
  
  
def getAngles(v1_array, v2_array = None, returnFullAngle = False):
  '''Inputs are:
  
      v1_array(samples x dim) - array of vectors (rows)
      v1_array(samples x dim) - second array of vectors; defaults to unit x-vector
      returnFullAngle (Bool)  - if True, returns [0, 2pi] angle measurement; else 
                                measures the inner angle between vectors (default)
      
      Returns array <angles> of angles in radians
  '''
    
  # normalize velocity vectors:
  v1_norms = np.linalg.norm(v1_array, axis = 1)
  v1_u     = np.divide(v1_array, v1_norms[:, np.newaxis])
  
  if v2_array is not None:
    v2_norms = np.linalg.norm(v2_array, axis = 1)
    v2_u     = np.divide(v2_array, v2_norms[:, np.newaxis])
  else:
    v2_u       = np.zeros(v1_u.shape) 
    v2_u[:, 0] = 1
  
  if returnFullAngle:
    assert v1_array.shape[1] == 2, "Only works for 2D vectors!"
    assert v2_array is None, "Only computed against reference (1, 0) unit vector."
    
    angles                        = np.arctan2(v1_u[:, 1], v1_u[:, 0]) 
    mask                          = np.zeros(angles.shape) 
    mask[np.where(angles < 0)[0]] = 2 * np.pi
    angles                       += mask        # correct for discontinuity at 180 deg where it flips to -180
  
  else: # cos(theta) = <a, b> / ||a|| ||b||  where a, b have norm = 1 here:
    angles   = np.arccos(np.clip(np.sum(v1_u * v2_u, axis = 1), -1.0, 1.0))
  
  return angles



def fit_BinnedAngles(datas, n_AngleBins, time_bin = 10):
  '''Converts continuous 2D vector signal into binned angles. Inputs are:
  
      datas (2D array)  - time x 2 array of cursor positions 
      n_AngleBins (int) - number of bins to chop [0, 2pi] angles into
      
    Returns:
      
      binned_angles (1D array) - sequence of ints specifying bin each time period 
                                 belongs to
      labels (1D array)        - value of each bin (avg angle covered)
  '''
  
  vel_angle     = getAngles(datas, returnFullAngle = True)
  
  AngleBins     = np.linspace(0, 2 * np.pi, n_AngleBins + 1)
  labels        = np.asarray([(AngleBins[i] + AngleBins[i]) / 2 for i in range(n_AngleBins)])
  binned_angles = np.digitize(vel_angle, AngleBins)
  
  return binned_angles, labels

def fitTuningCurve(neural, traj, n_AngleBins, percentile = 0.95, n_iters = 1000):
    '''Fit tuning curve data for a given unit's activity. Inputs are:

    neural (2D array) - neural activity (time x channels)
    traj (2D array)   - time x 2 of velocity/cursor signals
    n_AngleBins (int) - number of bins to divide [0, 2pi] into
    n_iters (int)     - number of bootstrap iterations 

    Returns:

      FR_estimates (2D array) - n_AngleBins x 3 array of FR mean estimates (mean, low, high)
      labels (1D array)       - corresponding angle value for each angle bin
    '''

    n_samples, n_channels  = neural.shape[:]
    FR_estimates           = np.zeros((n_AngleBins, n_channels, 3))
    binned, labels         = fit_BinnedAngles(traj, n_AngleBins)

    for i in range(n_AngleBins):       
        angle_idxs = np.where(binned == i+1)[0]
        
        FR_estimates[i, :, 0] = neural[angle_idxs, :].mean(axis = 0)
                
        bs = scipy.stats.bootstrap((neural[angle_idxs, :],), np.mean, axis = 0, 
                                   confidence_level = percentile, n_resamples = n_iters,
                                   method = 'percentile')
        FR_estimates[i, :, 1] = bs.confidence_interval.low
        FR_estimates[i, :, 2] = bs.confidence_interval.high

    return FR_estimates, labels

  
def fitCosineTuning(neural, traj):
  '''Fit cosine tuning model to neural and cursor data. Inputs are:
  
    neural (1D array) - time x 1 vector of unit activity
    traj (2D array)   - time x 2 array of velocity/error signals
  '''
  
  vel_angle = getAngles(traj, returnFullAngle = True)
  features  = np.vstack([np.ones(vel_angle.shape), np.cos(vel_angle), np.sin(vel_angle)]).T
  
  coefs     = np.linalg.lstsq(features, neural, rcond = None)[0]
  
  FR_0      = coefs[0]
  theta     = np.arctan2(coefs[2], coefs[1])
  A         = np.linalg.norm(coefs[1:])
  
  return FR_0, theta, A

  

class CosineTuningModel(object):
    """
    Cosine tuning model.
    """
    def __init__(self, fitMean = True):
        self.r_0     = None
        self.alpha   = None
        self.theta   = None
        self.fitMean = fitMean
        
    def fit(self, angles, neural):
        '''Fit cosine tuning model to neural and cursor data. Inputs are:

          angles (1D array)   - time x 1 array of angles
          neural (1D array) - time x 1 vector of unit activity'''
    
        if self.fitMean:
            features   = np.vstack([np.ones(angles.shape), np.cos(angles), np.sin(angles)]).T
            coefs      = np.linalg.lstsq(features, neural, rcond = None)[0]
            self.r_0   = coefs[0]
            self.alpha = np.linalg.norm(coefs[1:])
            self.theta = np.arctan2(coefs[2], coefs[1])
        else:
            features   = np.vstack([np.cos(angles), np.sin(angles)]).T
            coefs      = np.linalg.lstsq(features, neural, rcond = None)[0]
            self.r_0   = 0
            self.alpha = np.linalg.norm(coefs)
            self.theta = np.arctan2(coefs[1], coefs[0])
        return self
    
    def predict(self, angles):
        preds = self.r_0 + (self.alpha * np.cos(self.theta - angles))
        return preds

    def score(self, angles, neural):
        preds = self.predict(angles)
        score = r2_score(neural, preds)
        return score



def fitEncodingMatrix(cursorSig, neural, fitMean = True):
    '''Fit cosine tuning model to neural and cursor data. Inputs are:

          cursorSig (2D array) - time x 2 array of cursor signals
          neural (2D array)    - time x n_chans vector of unit activity
          fitMean (bool)       - whether or not to model baseline; set to False
                                 if using already centered data
     '''
    
    n_chans = neural.shape[1]
    enc     = np.zeros((3, n_chans))
    cm_list = list()
    mask    = np.linalg.norm(cursorSig, axis = 1) > 0
    
    for i in range(n_chans):
        
        angles = getAngles(cursorSig[mask, :], returnFullAngle = True)
        cm     = CosineTuningModel(fitMean = fitMean)
        
        cm.fit(angles, neural[mask, i])
        enc[0, i]  = cm.r_0
        enc[1:, i] = np.asarray([np.cos(cm.theta), np.sin(cm.theta)]) * cm.alpha
        cm_list.append(cm)
        
    return enc, cm_list


    
    
    
class FlexibleVonMisesModel(object):
    """
    Extension of standard cosine tuning model, with a set of tunable Von Mises
    functions to better model bimodal peaks, dispersion, etc. 
    
    TODO: add fitMean bool argument
    """
    def __init__(self, n_basis, kappa,):
        self.n_basis  = n_basis                                                # number of Von Mises functions to use
        self.vm_angle = np.linspace(0, 2 * np.pi, n_basis, endpoint = False)   # where each distn concentrated at 
        self.kappa    = kappa                                                  # dispersion parameter for these functions
        self.weights  = None                                                   # learnable weighting for each Von Mises fxn
        self.r_0      = None                                                   # avg FR (bias)
        
    
    def vm_prob(self, vm_num, x):
      '''Get (unnormalized) probability of angle(s) <x> under 
         one of the Von Mises distributions (specificed by vm_num).
      '''
      prob = np.exp(self.kappa * np.cos(self.vm_angle[vm_num] - x))
      return prob 
      
        
    def fit(self, angles, neural):
      '''Fit cosine tuning model to neural and cursor data. Inputs are:
          
          angles (1D array)   - time x 1 array of angles
          neural (1D array) - time x 1 vector of unit activity
      '''
      vm_features = ([self.vm_prob(i, angles) for i in range(self.n_basis)])
      
      features   = np.vstack([np.ones(angles.shape), vm_features]).T
      coefs      = np.linalg.lstsq(features, neural, rcond = None)[0]

      self.r_0     = coefs[0]
      self.weights = coefs[1:]
      
      return self
    
    def predict(self, angles):
      '''Predict neural activity based on angular measurements using:
          z_t | theta_t ~ N(r_0 + w^T [\phi_1, ..., phi_n]), Q)
      '''
      
      vm_probs = np.zeros((self.n_basis, len(angles)))
      for i in range(self.n_basis):
        vm_probs[i, :] = self.vm_prob(i, angles) * self.weights[i]
        
      marg  = np.sum(vm_probs, axis = 0)
      preds = self.r_0 + marg
      
      return preds
    
    
    def score(self, angles, neural):
      preds = self.predict(angles)
      score = r2_score(neural, preds)
      
      return score