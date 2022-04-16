import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage



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


def rolling_window(a, window_size, padding = None):
    '''Generate windowed features from a 2D timeseries. Inputs are:
    
        a (time x features) - array to segment 
        window_size (int)   - timelength of windows 
        
      Returns '''
    
    if padding is not None:
        zeros = np.zeros((window_size - 1, *a.shape[1:])) + padding
        a     = np.concatenate([zeros, a])
    
    shape   = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def plotRaster(raster_matrix, time_zero = 0):
    '''Make raster plot of spike trains in raster_matrix. Inputs are:

    raster_matrix (2D array) - neurons x time matrix of spikes
    time_zero (int)          - 0 label on plot 
    '''
    raise NotImplementedError
    # spike_times = list()

    # for unit in 1:raster_matrix.shape[0]
    #   spike_times[unit] = np.nonzero(raster_matrix[unit,:])[1]

    # raster_data = np.array
  


