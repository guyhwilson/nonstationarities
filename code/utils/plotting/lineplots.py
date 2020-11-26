import matplotlib.pyplot as plt
import numpy as np
import sys
from plotting_utils import figSize


def plotsd(data, color, time_bins = None, toggleSE = False, alpha = 0.2):
  '''
  Plotting with standard error shading. Inputs are:
  
  data (2D array or list)  - reps x time where SDs are taken along 1st axis;
                             else a list with entries containing reps for each point
  color                    - color value 
  time_bins (1D array)     - timestamp of bins from data 
  toggleSE (Boolean)       - if true, use SE shading 
  '''
  
  if isinstance(data, list):
    mean_signal = np.asarray([np.mean(i) for i in data])
    sd_signal   = np.asarray([np.std(i) for i in data])
    numreps     = np.asarray([len(i) for i in data])

  else:
    mean_signal = np.mean(data, axis = 0)
    sd_signal   = np.std(data, axis= 0) 
    numreps     = data.shape[0]
    
  if time_bins is None:
    time_bins = np.arange(0, data.shape[1])
  
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
  
  
def comparisonScatterPlot(x, y, figsize = 10, xy_lims = None):

	if xy_lims is None:
		xy_lims = list()
		xy_lims.append(min(np.min(x), np.min(y)))
		xy_lims.append(max(np.max(x), np.max(y)))

	figSize(figsize, figsize)
	scatter_ax = plt.scatter(x, y)
	unity_ax   = plt.plot(xy_lims, xy_lims, color = 'k', linestyle = '--')
	
	plt.xlim(xy_lims)
	plt.ylim(xy_lims)

	return scatter_ax, unity_ax

  