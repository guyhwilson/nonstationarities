import numpy as np


def generateUnits(n_units, SNR = 1):
	'''Generate PDs and baseline FRs for requested number of units.
	Inputs are: 
	
		n_units (int) - number of units to simulate
		SNR (float)   - norm of population tuning vector 
	
	Returns:
		
		tuning (2D array) - n_units x 3 array; 1st column has means 
		                    while last two are tuning for x and y vel.
		 
	We make sure that the last two columns are orthogonal (uniformly distributed PDs).
	'''
	
	tuning         = np.random.normal(size = (n_units, 3))
	tuning[:, 1:]  = np.linalg.qr(tuning[:, 1:], 'reduced')[0] * SNR
	
	return tuning


def simulateTuningShift(tuning, PD_ratio, mean_shift = 0):
	''' Simulate tuning shift for units. Inputs are:
	
		tuning (2D np array) - n_units x 3 array of tuning data
		PD_shift (float)     - relative strength of tuning in original PD
							   subspace following shift 
		mean_shift (float)   - strength of average mean change 
	''' 
	
	newTuning           = np.copy(tuning)
	newPD_component     = np.random.normal(size = (tuning.shape[0], 2))
	newPD_component, _  = np.linalg.qr(newPD_component, 'reduced')
	
	newTuning[:,1:]     = (tuning[:,1:] * PD_ratio) + (newPD_component *np.sqrt(1 - PD_ratio**2))
	newTuning[:, 0]    += np.random.normal(loc = 0, scale = mean_shift, size = tuning.shape[0])
	
	return newTuning


#def normalize_Decoder():
	
	

	

	


#def simulate_
	
	
	