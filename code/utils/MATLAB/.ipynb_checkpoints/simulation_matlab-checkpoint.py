import numpy as np
import matlab.engine


def engineArray2Python(array):
    '''Convert matlab.engine arrays into numpy in an efficient manner.
       You should basically always use this.'''
    
    py_arr = np.array(array._data).reshape(array.size, order='F')
    return py_arr



def MAT_simulateBCIFitts(neuralTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim, addpath = 'utils/MATLAB/', engine = None):
  
	if engine is None:
		eng  = matlab.engine.start_matlab()
	else:
		eng = engine

	eng.addpath('C:/Users/ghwilson/Documents/projects/nonstationarities/code/utils/MATLAB/')

	posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt = eng.simulateBCIFitts(matlab.double(neuralTuning.tolist()), matlab.double(D.tolist()), float(alpha), 
																										float(beta), float(nDelaySteps), float(delT), float(nStepsForSim), nargout = 8)
	if engine is None:
		eng.quit()

	return engineArray2Python(posTraj), engineArray2Python(velTraj), engineArray2Python(rawDecTraj), engineArray2Python(conTraj), engineArray2Python(targTraj), engineArray2Python(neuralTraj), engineArray2Python(trialStart), engineArray2Python(ttt)



def simulateBCI_KF(alpha, beta, nDelaySteps, delT, addpath = 'utils/MATLAB/', engine = None):
  
	if engine is None:
		eng  = matlab.engine.start_matlab()
	else:
		eng = engine

	eng.addpath('C:/Users/ghwilson/Documents/projects/nonstationarities/code/utils/MATLAB/')

	L_kalman, A_aug, B_aug, C_aug, D_aug, G_aug, H_aug = eng.SimulateBCI_KF(float(alpha), float(beta), int(nDelaySteps), float(delT), nargout = 7)
	if engine is None:
		eng.quit()

	return engineArray2Python(L_kalman), engineArray2Python(A_aug), engineArray2Python(B_aug), engineArray2Python(C_aug), engineArray2Python(D_aug), engineArray2Python(G_aug), engineArray2Python(H_aug)





def MAT_simulateBCIFitts_parallel(neural_list, D, alpha, beta, nDelaySteps, delT, nStepsForSim, addpath = 'utils/MATLAB/', engine = None):
	''' Simulate BCI Fitts task for multiple parameter sets at once.
	
	'''
	
	n_trls  = len(neural_list)
	
	if engine is None:
		eng = matlab.engine.start_matlab()
	else:
		eng = engine

	eng.addpath('C:/Users/ghwilson/Documents/projects/nonstationarities/code/utils/MATLAB/')

	#posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt = eng.simulateBCIFitts_parallel(neural_list, matlab.double(D.tolist()), float(alpha), 
	#																									float(beta), float(nDelaySteps), float(delT), float(nStepsForSim), nargout = 8)
	ttt = eng.simulateBCIFitts_parallel(neural_list, matlab.double(D.tolist()), float(alpha), float(beta), float(nDelaySteps), float(delT), float(nStepsForSim), nargout = 1)
	
	if engine is None:
		eng.quit()
	
	#posTraj     = [engineArray2Python(posTraj[i]) for i in range(n_trls)] 
	#velTraj     = [engineArray2Python(velTraj[i]) for i in range(n_trls)]
	#rawDecTraj  = [engineArray2Python(rawDecTraj[i]) for i in range(n_trls)]
	#conTraj     = [engineArray2Python(conTraj[i]) for i in range(n_trls)]
	#targTraj    = [engineArray2Python(targTraj[i]) for i in range(n_trls)]
	#neuralTraj  = [engineArray2Python(neuralTraj[i]) for i in range(n_trls)]
	#trialStart  = [engineArray2Python(trialStart[i]) for i in range(n_trls)]
	ttt         = [engineArray2Python(ttt[i]) for i in range(n_trls)]
		

	#return posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt 
	return ttt 


def generateDistortionMatrices_parallel(neural_list, D, alpha, beta, nDelaySteps, delT, nStepsForSim, addpath = 'utils/MATLAB/', engine = None):
	''' Simulate BCI Fitts task for multiple parameter sets at once.
	
	'''
	
	n_trls  = len(neural_list)
	
	if engine is None:
		eng = matlab.engine.start_matlab()
	else:
		eng = engine

	eng.addpath('C:/Users/ghwilson/Documents/projects/nonstationarities/code/utils/MATLAB/')
	
	dist_mats, ttt = eng.generateDistortionMatrices_parallel([matlab.double(neural_list[i].tolist()) for i in range(n_trls)], matlab.double(D.tolist()), float(alpha), float(beta), float(nDelaySteps), float(delT), float(nStepsForSim), nargout = 2)
	
	if engine is None:
		eng.quit()
	
	dist_mats    = [engineArray2Python(dist_mats[i]) for i in range(n_trls)]
	ttt          = [engineArray2Python(ttt[i]) for i in range(n_trls)]

	return dist_mats, ttt 

