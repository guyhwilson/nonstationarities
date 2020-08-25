import matplotlib.pyplot as plt
import numpy as np
import sys


def plotsd(data, color, time_bins = None, toggleSE = False, alpha = 0.2):
  '''
  Plotting with standard error shading. Inputs are:
  
  data (2D array)      - reps x time; SDs are taken along 1st axis
  color                - color value 
  time_bins (1D array) - timestamp of bins from data 
  toggleSE (Boolean)   - if true, use SE shading 
  '''
  
  numreps     = data.shape[0]
  if time_bins is None:
    time_bins = np.arange(0, data.shape[1])
  
  mean_signal = np.mean(data, axis = 0)
  sd_signal   = np.std(data, axis= 0) 
  
  if toggleSE:
    sd_signal /= np.sqrt(numreps)
  
  plt.plot(time_bins, mean_signal, color= color)
  plt.fill_between(time_bins, mean_signal - sd_signal, mean_signal + sd_signal, color=color, alpha=alpha)
  
  
def plotCI(data, color, CI = 95, time_bins = None, alpha = 0.2):
  '''
  Plotting with standard error shading. Inputs are:

  data (2D array)      - reps x time; CIs are taken along the first axis
  CI (float)           - percentiles to plot (default: 95%)
  color                - color value 
  time_bins (1D array) - timestamp of bins from data 
  toggleSE (Boolean)   - if true, use SE shading 
  '''

  numreps     = data.shape[0]
  if time_bins is None:
    time_bins = np.arange(0, data.shape[1])
  
  mean_signal = np.mean(data, axis = 0)
  flanking    = (100 - CI)/2
  up_q, low_q = 100 - flanking, flanking
  
  upper       = np.percentile(data, up_q, axis= 0) 
  lower       = np.percentile(data, low_q, axis = 0)
  
  
  plt.plot(time_bins, mean_signal, color= color)
  plt.fill_between(time_bins, lower, upper, color=color, alpha=alpha)
  
  
  