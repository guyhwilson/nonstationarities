import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sys
sys.path.append('utils/MATLAB/')
sys.path.append('utils/preprocessing/')
sys.path.append('utils/recalibration/')
from recalibration_utils import *
from session_utils import *
from HMM_matlab import *
from preprocess import DataStruct, daysBetween



def get_DiscreteTargetGrid(struct, gridSize, task = None):
	'''Divide screen into a n x n grid of possible target locations. Inputs are: 

	struct (DataStruct) - session data to use 
	gridSize (int)      - number of rows and columns to chop the screen up into 
	task (str)          - task data to draw from; defaults to using all 

	TODO: update to work with new getTrainTest outputs, also deal with screen shifting around for different blocks
	'''

	if task is None:
		targpos_data = struct.targetPos_continuous
	else:
		targpos_data = np.concatenate([struct.targetPos[i] for i in np.where(struct.trialType == task)[0]])

	X_min, X_max  = targpos_data[:, 0].min() - 20, targpos_data[:, 0].max() + 20
	Y_min, Y_max  = targpos_data[:, 1].min() - 20, targpos_data[:, 1].max() + 20

	X_loc,Y_loc   = np.meshgrid(np.linspace(X_min, X_max, gridSize), np.linspace(Y_min, Y_max, gridSize))
	targLocs      = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T

	return targLocs 



def prep_HMMData(struct, train_frac = 1., task = None, blocks = None, cutStart = None, return_flattened = False):
	'''
	Code for generating input data for HMM using session data. Inputs are:

	struct (DataStruct)      - session to train on
	train_frac (float)       - fraction of dataset to use on training 
	task (str)               - task type to train and test on
	blocks (str)             - blocks to use 

	Returns:

	train/test_neural    - 
	train/test_cursorPos - 
	train/test_cursorErr - 

	TODO: 
	- join adjacent trials so that the returned list contains contiguous segments
	- figure out if returning individual trial lists to train_HMMRecalibrate causes bad performance (linear reg
	  models show bad performance because the time snippets are so short)
	- maybe trash and just add optional return_cursorPos parameter to getTrainTest()
	'''
		
	neural, cursorErr, targPos       = getNeuralAndCursor(struct, sigma = None, task = task, blocks = blocks)
	
	n_trls                           = len(neural)
	train_ind, test_ind              = train_test_split(np.arange(n_trls), train_size = train_frac, shuffle = False)
	
	if cutStart is not None:
		neural    = [neural[i][cutStart:, :] for i in range(n_trls)]
		cursorErr = [cursorErr[i][cutStart:, :] for i in range(n_trls)]

	train_neural, test_neural        = [neural[i] for i in train_ind], [neural[i] for i in test_ind]
	train_cursorErr, test_cursorErr  = [cursorErr[i] for i in train_ind], [cursorErr[i] for i in test_ind]
	train_cursorPos, test_cursorPos  = [-1 * (cursorErr[i] - targPos[i]) for i in train_ind], [-1 * (cursorErr[i] - targPos[i]) for i in test_ind]
	
	if return_flattened:
		train_neural    = np.vstack(train_neural)
		test_neural     = np.vstack(test_neural)
		train_cursorPos = np.vstack(train_cursorPos)
		test_cursorPos  = np.vstack(test_cursorPos)
		train_cursorErr = np.vstack(train_cursorErr)
		test_cursorErr  = np.vstack(test_cursorErr)

	
	return train_neural, train_cursorPos, train_cursorErr, test_neural, test_cursorPos, test_cursorErr



def HMM_estimate(decOutput, cursorPos, stateTrans, pStateStart, targLocs, vmKappa, vmAdjust = [0.1, 40.], engine = None):
	'''
	Code for training linear decoder on session data using HMM-inferred target locations. Inputs are:

		decOutput (list)         - entries are time x 2 arrays of cursor estimates from decoder
		cursorPos (list)         - entries are time x 2 arrays of cursor positions
		stateTrans (2D array)    - transition probability matrix for HMM
		pStatestart (1D array)   - probability of beginning in a given state
		targLocs (2D arrays)     - corresponding locations on grid for targets
		vmKappa (float)          - precision parameter for the von mises distribution.
		vmAdjust (float tuple)   - parameters for kappa adjustment in HMM; terms are [a, b] 
								   where K_adj = K * 1 / ( 1 + exp(-(distance - a) * b  ))
	'''
	assert len(decOutput) == len(cursorPos), "Different number of decoder output trials and cursor position trials"
	
	n_trials               = len(decOutput)
	targStates, pTargState = list(), list()
	
	for i in range(n_trials):
		targs      = hmmviterbi_vonmises(decOutput[i], stateTrans, targLocs, cursorPos[i], pStateStart, vmKappa, vmAdjust = vmAdjust, engine = engine)[0]
		pTargs     = hmmdecode_vonmises(decOutput[i], stateTrans, targLocs, cursorPos[i], pStateStart, vmKappa, vmAdjust = vmAdjust, engine = engine)[0]
		targStates.append(targs)
		pTargState.append(pTargs)
	
	targStates  = np.concatenate(targStates)
	pTargState  = np.concatenate(pTargState)

	return targStates, pTargState
	



def train_HMMRecalibrate(decoder, neural, cursorPos, stateTrans, pStateStart, targLocs, vmKappa, probThreshold = 0.):
	'''
	Code for training linear decoder on session data using HMM-inferred target locations. Inputs are:

		decoder (Sklearn object) - decoder to use 
		neural (list)            - entries are time x channels arrays of neural activity
		cursorPos (list)         - entries are time x 2 arrays of cursor positions
		stateTrans (2D array)    - transition probability matrix for HMM
		pStatestart (1D array)   - probability of beginning in a given state
		targLocs (2D arrays)     - corresponding locations on grid for targets
		vmKappa (float)          - precision parameter for the von mises distribution.
		probThreshold (float)    - threshold for subselecting high certainty regions (only where best guess > probThreshold)
	'''

	n_trials               = len(neural)
	targStates, pTargState = list(), list()

	neural_flattened       = np.concatenate(neural)
	cursorPos_flattened    = np.concatenate(cursorPos)
	
	for i in range(n_trials):
		rawDecTraj = decoder.predict(neural[i] - neural_flattened.mean(axis = 0))
		targs      = hmmviterbi_vonmises(rawDecTraj, stateTrans, targLocs, cursorPos[i], pStateStart, vmKappa)[0]
		pTargs     = hmmdecode_vonmises(rawDecTraj, stateTrans, targLocs, cursorPos[i], pStateStart, vmKappa)[0]
		targStates.append(targs)
		pTargState.append(pTargs)
	
	targStates  = np.concatenate(targStates)
	pTargState  = np.concatenate(pTargState)

	maxProb     = np.max(pTargState, axis = 0)              
	highProbIdx = np.where(maxProb > probThreshold)[0]                   # find time periods of high certainty
	
	inferredTargLoc = targLocs[targStates.astype('int').flatten() - 1,:] # find predicted target locations for high prob times
	inferredPosErr  = inferredTargLoc - cursorPos_flattened              # generate inferred cursorErr signals

	# if data too noisy or sessions too far apart, HMM may not have any valid high confidence time points
	# so we decrement the threshold by 0.1 until valid samples exist 
	isFitted        = False
	while not isFitted:
		try:
			decoder.fit(neural_flattened[highProbIdx, :] - neural_flattened.mean(axis = 0), inferredPosErr[highProbIdx, :])
			isFitted = True
		except:
			probThreshold -= 0.1
			highProbIdx    = np.where(maxProb > probThreshold)[0]
			print('ProbThreshold too high. Lowering by 0.1')

	return decoder

