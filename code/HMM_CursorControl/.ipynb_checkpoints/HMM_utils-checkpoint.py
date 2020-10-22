import numpy as np
import matlab.engine




def simulateBCIFitts(neuralTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim, addpath = None):
  
    eng = matlab.engine.start_matlab()
    posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt = eng.simulateBCIFitts(matlab.double(neuralTuning.tolist()), matlab.double(D.tolist()), float(alpha), 
                                                                                                        float(beta), float(nDelaySteps), float(delT), float(nSimSteps), nargout = 8)
    eng.quit()
    return posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt

  
def hmmviterbi_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, pStateStart, vmKappa, addpath = None):
    eng              = matlab.engine.start_matlab()
    targStates, logP = eng.hmmviterbi_vonmises(matlab.double(rawDecodeVec.tolist()), matlab.double(stateTransitions.tolist()), matlab.double(targLocs.tolist()), 
                                               matlab.double(cursorPos.tolist()), matlab.double(pStateStart.tolist()), float(vmKappa), nargout = 2)
    eng.quit()
    
    return targStates, logP
    
    
def hmmdecode_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, pStateStart, vmKappa, addpath = None):
    eng              = matlab.engine.start_matlab()
    pTargState, pSeq = eng.hmmdecode_vonmises(matlab.double(rawDecodeVec.tolist()), matlab.double(stateTransitions.tolist()), matlab.double(targLocs.tolist()), 
                                               matlab.double(cursorPos.tolist()), matlab.double(pStateStart.tolist()), float(vmKappa), nargout = 2)
    eng.quit()
    
    return pTargState, pSeq