import numpy as np
import matlab.engine
from simulation_matlab import engineArray2Python


  
def hmmviterbi_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, pStateStart, vmKappa, addpath = 'MATLAB/', engine = None):
	'''
	Wrapper function for calling matlab HMM code. Can optionally pass a pre-initialized matlab.engine object for speeding up sequential MATLAB calls 
	(bypass time required to start each engine call).
	'''
	
	if engine is None:
		eng  = matlab.engine.start_matlab()
	else:
		eng = engine

	eng.addpath('C:/Users/ghwilson/Documents/projects/nonstationarities/code/utils/MATLAB/')
	targStates, logP = eng.hmmviterbi_vonmises(matlab.double(rawDecodeVec.tolist()), matlab.double(stateTransitions.tolist()), matlab.double(targLocs.tolist()), 
											   matlab.double(cursorPos.tolist()), matlab.double(pStateStart.tolist()), float(vmKappa), nargout = 2)
	if engine is None:
		eng.quit()

	return engineArray2Python(targStates), logP

    
def hmmdecode_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, pStateStart, vmKappa, addpath = 'MATLAB/', engine = None):
	'''
	Wrapper function for calling matlab HMM code. Can optionally pass a pre-initialized matlab.engine object for speeding up sequential MATLAB calls 
	(bypass time required to start each engine call).
	'''
	
	if engine is None:
		eng  = matlab.engine.start_matlab()
	else:
		eng = engine

	eng.addpath('C:/Users/ghwilson/Documents/projects/nonstationarities/code/utils/MATLAB/')
	pTargState, pSeq = eng.hmmdecode_vonmises(matlab.double(rawDecodeVec.tolist()), matlab.double(stateTransitions.tolist()), matlab.double(targLocs.tolist()), 
											   matlab.double(cursorPos.tolist()), matlab.double(pStateStart.tolist()), float(vmKappa), nargout = 2)
	if engine is None:
		eng.quit()

	return engineArray2Python(pTargState), pSeq