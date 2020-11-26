import numpy as np
import matlab.engine


def engineArray2Python(array):
    '''Convert matlab.engine arrays into numpy in an efficient manner.
       You should basically always use this.'''
    
    py_arr = np.array(array._data).reshape(array.size, order='F')
    return py_arr



def simulateBCIFitts(neuralTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim, addpath = 'utils/MATLAB/'):
  
    eng = matlab.engine.start_matlab()
   # eng.addpath(addpath)
    eng.addpath('C:/Users/ghwilson/Documents/projects/nonstationarities/code/utils/MATLAB/')
    
    posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt = eng.simulateBCIFitts(matlab.double(neuralTuning.tolist()), matlab.double(D.tolist()), float(alpha), 
                                                                                                        float(beta), float(nDelaySteps), float(delT), float(nStepsForSim), nargout = 8)
    eng.quit()
    return engineArray2Python(posTraj), engineArray2Python(velTraj), engineArray2Python(rawDecTraj), engineArray2Python(conTraj), engineArray2Python(targTraj), engineArray2Python(neuralTraj), engineArray2Python(trialStart), engineArray2Python(ttt)