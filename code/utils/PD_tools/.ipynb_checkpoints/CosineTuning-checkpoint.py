import numpy as np
#from sklearn.linear_model import Lin


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
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
  
  
def getAngles(v1_array, v2_array):
  '''Inputs are:
  
      v1_array(samples x dim) - array of vectors (rows)
      v1_array(samples x dim) - second array of vectors
  '''
    
  # normalize velocity vectors:
  v1_norms = np.linalg.norm(v1_array, axis = 1)
  v1_u     = np.divide(v1_array, v1_norms[:, np.newaxis])
  
  v2_norms = np.linalg.norm(v2_array, axis = 1)
  v2_u     = np.divide(v2_array, v2_norms[:, np.newaxis])
  
  # cos(theta) = <a, b> / ||a|| ||b||  where a, b have norm = 1 here:
  angles   = np.arccos(np.clip(np.sum(v1_u * v2_u, axis = 1), -1.0, 1.0))
  
  return angles

                       
  


class CosineModel(object):
    """
    Cosine tuning model.
    """
    def __init__(self):
        self.r_0   = None
        self.r_max = None
        self.theta = None
        
    def fit(x, y):
      '''
      Takes input 
      '''
      
      
      