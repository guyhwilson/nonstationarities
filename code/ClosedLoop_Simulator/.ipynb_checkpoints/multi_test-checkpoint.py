import numpy as np
import sys, glob
[sys.path.append(f) for f in glob.glob('utils/*')]

from hmm_utils import *
from simulation_matlab import *
from simulation_utils import * 


def myFunc(i):
    seq = [0, 1]
    for j in range(100000):
        seq.append(seq[-1] + seq[-2])
    return seq[0]


def getDistortionMatrix(dec, enc):
    '''
    Get distortion matrix D^T * E. Inputs are:
    
        dec (2D array) - n_channels x 2 array
        enc (2D array) - n_channels x 2 array 
    '''
    
    distort  = dec.T.dot(enc)
    distort /= np.outer(np.linalg.norm(dec, axis = 0), np.linalg.norm(enc, axis = 0))
    
    return distort



def generateDistortionMatrices(parallel_args):
	D, initialTuning, days_between, tuning_shift, alpha, beta, nDelaySteps, delT, nSimSteps, eng = parallel_args

	D_new  = np.copy(D)
	tuning = np.copy(initialTuning)
	for k in range(days_between + 1):
		tuning = simulateTuningShift(tuning, PD_ratio = tuning_shift[0], mean_shift = tuning_shift[1])  # introduce daily nonstationarities between recorded sessions

	# simulate new days' data:
	cursorPos, _, decOut, _, targetPos, neural, _, ttt = simulateBCIFitts(tuning, D_new, alpha, beta, nDelaySteps, delT, nSimSteps, engine = eng) 

	PosErr     = targetPos - cursorPos  
	D_new      = np.linalg.lstsq(np.hstack([np.ones((neural.shape[0], 1)), neural]), PosErr, rcond = -1)[0]  # update previous decoder
	decVec_new = np.hstack([np.ones((neural.shape[0], 1)), neural]).dot(D_new)

	#Important: normalize the decoder so that D_new decoders vectors of magnitude 1 when far from the target. 
	#This will restore the original optimal gain.
	TargDist   = np.linalg.norm(PosErr, axis = 1)
	TargDir    = PosErr / TargDist[:, np.newaxis]
	farIdx     = np.where(TargDist > 0.4)[0]
	projVec    = np.sum(np.multiply(decVec_new[farIdx, :], TargDir[farIdx, :]), axis = 1)
	D_new     /= np.mean(projVec)

	ttt_new                  = simulateBCIFitts(tuning, D_new, alpha, beta, nDelaySteps, delT, nSimSteps, engine = eng)[-1] # simulate performance with recalibrated decoder
	distortion               = getDistortionMatrix(D_new[1:, :], initialTuning[:, 1:])

	return distortion, ttt_new