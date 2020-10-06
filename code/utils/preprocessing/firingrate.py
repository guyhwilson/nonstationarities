import numpy as np 
import matplotlib.pyplot as plt


def rolling_window(a, window):
  shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
  strides = a.strides + (a.strides[-1],)
  return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def plotRaster(raster_matrix, time_zero = 0):
  '''
  Make raster plot of spike trains in raster_matrix. Inputs are:
  
  raster_matrix (2D array) - neurons x time matrix of spikes
  time_zero (int)          - 0 label on plot 
  '''
  assert False, "Not implemented yet."
  return 
 # spike_times = list()
  
 # for unit in 1:raster_matrix.shape[0]
 #   spike_times[unit] = np.nonzero(raster_matrix[unit,:])[1]
  
 # raster_data = np.array
  

def raster2FR(data_struct, binsize = 20, overlap = None, toggle_HLFP = False):
  '''
  Process FRs for a DataStruct object. Inputs are:
  
    data_struct (Struct)     - simplified R_struct converted by Mat2Python
    binsize (int)            - FR bin size 
    overlap (int)            - overlap between adjacent bins; for now is binsize - 1 
    toggle_HLFP (Bool)       - if True, perform on high frequency LFP data 
  
  TO-DO:
    - enable general overlap values 
    - speed optimization 
    
  '''
  if overlap is None:
    overlap = binsize - 1
  
  assert overlap == binsize - 1, "only 1 ms offset currently supported."
  
  if toggle_HLFP:
    spikeraster = data_struct.HLFP.copy()
    source      = 'HLFP'
  else:
    spikeraster = data_struct.TX.copy()
    source      = 'spikeRaster'
  
  total_len = spikeraster.shape[0]
  numunits  = spikeraster.shape[1]
  FRdata    = np.zeros((total_len - binsize + 1, numunits))
  
  for unit in range(numunits):
    FRdata[:, unit] = np.convolve(spikeraster[:, unit] , np.ones(binsize), 'valid')
    
  setattr(data_struct, 'FRsource', source)  # keep track of underlying signal source
  setattr(data_struct, 'FRbinsize', binsize)
  setattr(data_struct, 'FRbinoverlap', overlap)
  setattr(data_struct, 'FR', FRdata)
  
  
'''

def clipSignal(data_struct, clip, clip_signal):
  \'''
  data_struct (DataStruct) - data to clip
  clip (float)             - multiplier 
  clip_signal (str)        - can be 'HLFP' or 'FR'
  \'''
  
  numunits = data_struct.HLFP[0].shape[1]
  numtrls  = len(data_struct.speechLabel)
  
  assert clip_signal == 'FR' or clip_signal == 'HLFP', "Signal source not recognized. Check <clip_signal> input."
  assert clip > 0,                                     "Invalid clip value (must > 0)"
  
  for i in range(numunits):
    channel_data = getTrialActivity(data_struct, units = [i], alignment = None, numbins = None, source = clip_signal)[0].ravel()
    clip_lim     = np.median(channel_data) * clip
    #print(np.median(channel_data), clip)
    for j in range(numtrls):
      if clip_signal == 'HLFP':
        data_struct.HLFP[j][data_struct.HLFP[j] > clip_lim] = clip_lim
      if clip_signal == 'FR':
        data_struct.FR[j][data_struct.FR[j] > clip_lim] = clip_lim
  
  print(clip_signal, ' clipped at ', clip, ' X above median.')
  
  
  
  
def getTrialActivity(data_struct, units = None, alignment = None, numbins = None, source= 'FR'):
  \'''
  Concatenate activity across trials for a given channel(s), shortening
  trials to length of the minimum one. Inputs are:
  
  data_struct (DataStruct) - dataset to use
  units (array)            - channels to analyze (default: all)
  alignment (array)        - timestamps to align trials to (default: 0)
  numbins (int)            - if not None, extract [numbins] bins after alignment time

  Returns:
    
    trl_data (list)        - length = len(neurons); holds trials x time FR data 
  \'''
  
  numunits    = data_struct.HLFP[0].shape[1]
  numtrls     = len(data_struct.speechLabel)
  
  assert source == 'FR' or source == 'HLFP' or source == 'spikeRaster', "Requested activity type not recognized (must be 'FR', \'spikeRaster\', or \'HLFP\')"
  
  sourcedata  = getattr(data_struct, source)
  if source  == 'spikeRaster' or source == 'HLFP':
    sourcedata = [sourcedata[i].T.copy() for i in range(numtrls)]
  
  if units is None:
    units     = np.arange(numunits)    # if not provided, perform for all recorded units 
  if alignment is None:
    alignment = np.zeros((numtrls,))   # if not provided, we assume are already aligned (e.g. to pre-trial start)
  if numbins is None:  
    trl_stops = [np.shape(l)[1] for l in sourcedata]
    trlens    = [int(stop - start) for stop, start in zip(trl_stops, alignment)]
    numbins   = min(trlens)
    
  trl_data    = list()
  for unit in units:
      stacked_trls = np.zeros((numtrls, numbins))
      for i,j in enumerate(sourcedata):
          trl_stretch        = np.arange(alignment[i], (numbins + alignment[i])).astype('int')
          stacked_trls[i, :] = j[unit, trl_stretch].copy() 
      trl_data.append(stacked_trls)
      
  return trl_data


def getChannelsTimeTrials(data_struct, units = None, alignment = None, numbins = 100, returnMean = False, source = 'FR'):
  \'''
  For selected units, get FR data across trials. Inputs are: 

  data_struct (R struct)    - structure to generate data from 
  units (iterable of ints)  - list of units to extract data for (defaults to all)
  alignment (array)         - timestamps to align trials to (default: 0)
  numbins (int)             - extract [numbins] bins after alignment time; average
                              over this window (default: 100 msec)
  returnMean (Bool)         - if True, avg over time window and return mean FRs (default: False)
  source (str)              - 'FR', 'spike', or 'HLFP' (default: 'FR')
  
  TODO:
    - maybe bother changing name to UnitsTimeTrials and fix every mention in code
    - on second thought, maybe dont do that unless stuck on desert island 
    - probably should standardize my language at *some* point though 
  \'''

  num_trls   = len(data_struct.speechLabel)
  units_data = getTrialActivity(data_struct, units = units, alignment = alignment, numbins = numbins, source = source)

  if returnMean:  # return averaging FR during selected time window 
    FR_data  = np.zeros((len(units_data), num_trls))
  else:           # return whole time course during windows; 
    FR_data  = np.zeros((len(units_data), numbins, num_trls))

  for unit in range(len(units_data)):
    if returnMean: 
      FR_data[unit, :]    = np.mean(units_data[unit], axis = 1)
    else:           
      FR_data[unit, :, :] = units_data[unit].transpose()

  return FR_data


  
def activityThreshold(data_struct, FR_threshold = 1, fraction_below = 0.1):
  \'''
  Given a DataStruct object, remove all data corresponding to low-firing units
  as determined by threshold parameter. Input values are: 
  
  data_struct (DataStruct) - dataset to be pruned (use FR field)
  FR_threshold (int)       - freq (Hz) to be used as a highpass filter; all units
                             that spike < 1 Hz in grand-averaged timecourse discarded.
  fraction_below (float)   - percentage of time FR can drop below threshold 
  \'''

  assert data_struct is not None, "Provide valid input structure."
  

  sample_period = data_struct.FRbinsize / 1000
  numunits      = data_struct.FR[0].shape[0]
  numtrls       = len(data_struct.speechLabel)
  stacked_trls  = getTrialActivity(data_struct)
  trlen         = stacked_trls[0].shape[1] 
  
  keep          = list()
  cleaned_spike = list()
  cleaned_HLFP  = list()
  cleaned_FR    = list()
  
  
  for i in range(0, numunits):
    FR_GA     = np.mean(stacked_trls[i], axis = 0) / sample_period   # convert to Hz
    lowfiring = np.where(FR_GA < FR_threshold)[0].shape[0] / trlen   # percentage of trial time where FR < 1 Hz
    
    if lowfiring <= fraction_below:   # if channel drops below threshold greater than 10% of time, reject it
      keep.append(int(i))
      
  keep = np.asarray(keep)
  print('Units thresholded. Pruning activity data...')
  #print(keep)
  for trl in range(numtrls):
    cleaned_spike.append(data_struct.spikeRaster[trl][:, keep].copy())
    cleaned_HLFP.append(data_struct.HLFP[trl][:, keep].copy())
    cleaned_FR.append(data_struct.FR[trl][keep, :].copy())
    
    # reduce memory constraints 
    data_struct.HLFP[trl]        = 0
    data_struct.spikeRaster[trl] = 0
    data_struct.FR[trl]          = 0
      
  setattr(data_struct, 'spikeRaster', cleaned_spike)
  setattr(data_struct, 'HLFP', cleaned_HLFP)
  setattr(data_struct, 'FR', cleaned_FR)
  setattr(data_struct, 'kept', keep.copy())
  
  print('Removed the following channels: ', np.setdiff1d(np.arange(numunits), keep))
  print('Done. Check struct attributes.')
  
  
  
def pruneUnits(data_struct, bad_units):
 \'''
  Given a DataStruct object, remove bad units supplied by user. Input values are: 
  
  data_struct (DataStruct) - dataset to be pruned
  bad_units (int iterable) - position IDs of units to be removed from data_struct

  \'''
  
  numunits      = data_struct.FR[0].shape[0]
  
  assert data_struct is not None,   "Provide valid input structure."
  assert max(bad_units) < numunits, "Struct cannot be indexed by supplied list."
  
  numtrls       = len(data_struct.speechLabel)
  keep          = np.setdiff1d(np.arange(numunits), bad_units)
  
  cleaned_spike = list()
  cleaned_HLFP  = list()
  cleaned_FR    = list()
 
  for trl in range(numtrls):
    try:
      cleaned_spike.append(data_struct.spikeRaster[trl][:, keep].copy())
    except:
      pass
    cleaned_HLFP.append(data_struct.HLFP[trl][:, keep].copy())
    cleaned_FR.append(data_struct.FR[trl][keep, :].copy())
    
    # reduce memory constraints 
    try:
      data_struct.spikeRaster[trl] = 0
    except:
      pass
    data_struct.HLFP[trl]        = 0
    data_struct.FR[trl]          = 0
      
  setattr(data_struct, 'HLFP', cleaned_HLFP)
  try:
    setattr(data_struct, 'spikeRaster', cleaned_spike)
  except:
    pass
  
  setattr(data_struct, 'FR', cleaned_FR)
  setattr(data_struct, 'kept', keep.copy())
  print('Pruned! Check struct.')
  
  
  
def pruneTrials(data_struct, bad_trls):
  \'''
  Given a DataStruct object, remove bad trials supplied by user. Input values are: 
  
  data_struct (DataStruct)  - dataset to be pruned
  bad_trials (int iterable) - trials to be removed from data_struct
  \'''
  
  numtrls      = len(data_struct.speechLabel)
  
  assert data_struct is not None,   "Provide valid input structure."
  assert max(bad_trls) < numtrls,  "Fewer trials total than provided list"
  
  stacked_trls  = getTrialActivity(data_struct)
  keep          = np.setdiff1d(np.arange(numtrls), bad_trls)
  
  cleaned_spike = list()
  cleaned_HLFP  = list()
  cleaned_FR    = list()
  
  cleaned_phonemes = list()
  cleaned_trlabels = list()
  cleaned_starts   = list()
  cleaned_stops    = list()
  
  for trl in keep:
    cleaned_spike.append(data_struct.spikeRaster[trl].copy())
    cleaned_HLFP.append(data_struct.HLFP[trl].copy())
    cleaned_FR.append(data_struct.FR[trl].copy())
    
    cleaned_phonemes.append(data_struct.phonemes[trl].copy())
    cleaned_trlabels.append(data_struct.speechLabel[trl].copy())
    cleaned_starts.append(data_struct.starts[trl].copy())
    cleaned_stops.append(data_struct.stops[trl].copy())
    
    
    # reduce memory constraints 
    data_struct.spikeRaster[trl] = 0
    data_struct.HLFP[trl]        = 0
    data_struct.FR[trl]          = 0
      
  setattr(data_struct, 'HLFP', cleaned_HLFP)
  setattr(data_struct, 'spikeRaster', cleaned_spike)
  setattr(data_struct, 'FR', cleaned_FR)
  setattr(data_struct, 'kept_trials', keep.copy())
  
  setattr(data_struct, 'speechLabel', cleaned_trlabels)
  setattr(data_struct, 'phonemes', cleaned_phonemes)
  setattr(data_struct, 'starts', cleaned_starts)
  setattr(data_struct, 'stops', cleaned_stops)
  
  print('Pruned! Check struct.')
  
  


def estimateBaseline(data_struct, neurons = None, toggleBootstrap = False, returnTrialWise = False, additional_offset = 0):
  \'''
  Following speech end time estimation (see segmentation.py), use pooled
  intertrial periods to estimate basal firing for a given unit. Inputs are: 
  
  data_struct (DataStruct)  - dataset to draw from 
  neurons (int array)       - neuron to process
  toggleBootstrap (Bool)    - if True, use bootstrapping to estimate sample mean
                              (use 10,000 iterations)
  returnTrialWise (Bool)    - if True, return neurons x trls matrix of trial-level 
                              baseline FRs
  additional_offset (int)   - measure baseline starting from this long after speech stop;
                              measured in msec 
                
  \'''
  if neurons is None:
    neurons   = np.arange(0, data_struct.FR[0].shape[0])
    
  numneurons  = len(neurons)
  numtrls     = len(data_struct.speechLabel)
  
  align_times = [additional_offset + x for x in data_struct.FRspeechStop.copy()]
  end_aligned = getTrialActivity(data_struct, neurons = neurons, alignment = align_times)
  
  if returnTrialWise:
    if toggleBootstrap:
        print('Not available for trial-level.')
      
    means     = np.zeros((numneurons, numtrls))
    sd        = np.zeros((numneurons, numtrls))
  else:
    means     = np.zeros((numneurons,))
    sd        = np.zeros((numneurons,))
  
  for neuron in range(0, numneurons):
    
    if returnTrialWise:
      means[neuron, :] = np.mean(end_aligned[neuron], axis = 1)
      sd[neuron, :]    = np.std(end_aligned[neuron], axis  = 1)
      
    else:
      flattened    = end_aligned[neuron].flatten()
      if toggleBootstrap:
        sampling   = 500
        bootstraps = np.zeros((10000,1))

        for i in range(0, bootstraps.shape[0]):
          sampled       = np.random.choice(flattened, (sampling, 1), replace = True)
          bootstraps[i] = np.mean(sampled)
        means[neuron] = np.mean(bootstraps) 
        sd[neuron]    = np.std(bootstraps)

      else:
        means[neuron] = np.mean(flattened)
        sd[neuron]    = np.std(flattened)
  
  return means, sd
  '''