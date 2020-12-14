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





def orthogonalizeAgainst(v2, v1):
	'''
	Orthogonalize v2 against vector v1, keeping this latter vector the same
	'''
	
	u2  = v2 - (v2.dot(v1) * v1)
	u2 /= np.linalg.norm(u2)
	
	return u2



def simulateTuningShift(tuning, PD_ratio, mean_shift = 0):
	''' Simulate tuning shift for units. Inputs are:
	
		tuning (2D np array) - n_units x 3 array of tuning data
		PD_shift (float)     - relative strength of tuning in original PD
							   subspace following shift 
		mean_shift (float)   - strength of average mean change 
		
	''' 
	
	newTuning             = np.copy(tuning)
	newPD_component       = np.random.normal(size = (tuning.shape[0], 2))
	newPD_component, _    = np.linalg.qr(newPD_component, 'reduced')
	newPD_component[:, 0] = orthogonalizeAgainst(newPD_component[:, 0], tuning[:, 1]) 
	newPD_component[:, 1] = orthogonalizeAgainst(newPD_component[:, 1], tuning[:, 2]) 
	
	newTuning[:,1:]     = (tuning[:,1:] * PD_ratio) + (newPD_component *np.sqrt(1 - PD_ratio**2))
	newTuning[:, 0]    += np.random.normal(loc = 0, scale = mean_shift, size = tuning.shape[0])
	
	return newTuning



def generateTargetGrid(gridSize, x_bounds = [-0.5, 0.5], y_bounds = [-0.5, 0.5]):
	'''
	Generate target grid for simulator.
	'''
	
	X_loc,Y_loc = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], gridSize), np.linspace(y_bounds[0], y_bounds[1], gridSize))
	targLocs    = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T
	
	return targLocs

def generateTransitionMatrix(gridSize, stayProb):
	'''
	Generate transition probability matrix for simulator targets.
	'''
	nStates     = gridSize**2
	stateTrans  = np.eye(nStates)*stayProb # Define the state transition matrix

	for x in range(nStates):
		idx                = np.setdiff1d(np.arange(nStates), x)
		stateTrans[x, idx] = (1-stayProb)/(nStates-1)

	pStateStart = np.zeros((nStates,1)) + 1/nStates
	
	return stateTrans, pStateStart
	
	



#def normalize_Decoder():
	
	

	

	


#def simulate_
	
	
	