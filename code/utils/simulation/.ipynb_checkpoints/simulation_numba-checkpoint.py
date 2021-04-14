from numba import jit
import numpy as np

import sys
sys.path.append('../utils/simulation/')
from simulation import getBCIFittsKF_py


@jit(nopython=True) 
def simulate_numba(L_kalman, A_aug, B_aug, C_aug, d_aug, g_aug, h_aug, neuralTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim):
    ''' Simulate BCI Fitts task with dwell target logic. Inputs are: 
    
            neuralTuning (2D array) - channels x 3 array holding baseline FR, X-velocity tuning, and Y-velocity tuning
            D (2D array)            - (channnels x 1) x 2 array for cursor decoder
            alpha (float)           - cursor decoder (time) smoothing
            beta (float)            - cursor decoder gain
            nDelaySteps (int)       - delay from visual feedback
            delT (float)            - amount of time per step (ms)
            nStepsForSim (int)      - timesteps to simulate
                                      
    Here we define the fTarg function for the simulated user's control policy, which
    is basically just a saturating nonlinearity (see the paper
    'Principled BCI decoder design and parameter selection using a
    feedback control model' for more details).
    '''
    
    fTarg = np.array([[0.00000000e+00, 3.43250000e-04, 2.74175000e-02, 1.02654500e-01,
        1.72120750e-01, 2.61818500e-01, 3.77416250e-01, 5.11568000e-01,
        6.59500750e-01, 8.23845250e-01, 9.99644250e-01, 1.00000000e+02],
       [0.00000000e+00, 0.00000000e+00, 5.00624667e-01, 6.02977667e-01,
        8.33672667e-01, 8.34671333e-01, 8.68126333e-01, 9.07594000e-01,
        9.36572667e-01, 1.03224767e+00, 9.97908667e-01, 9.97908667e-01]]).T


    # This is the main simulation loop.
    # These are some task parameters.
    nHoldSteps = 50
    holdCounter = 0
    trialCounter = 0
    maxSteps = 1000
    targRad = 0.075

    # Define matrices to hold the time series data.
    posTraj    = np.zeros((nStepsForSim, 2))
    posErr_est = np.zeros((nStepsForSim, 2))              # internal estimate of position error
    neuralTraj = np.zeros((nStepsForSim, neuralTuning.shape[0]))
    velTraj    = np.zeros((nStepsForSim, 2))
    rawDecTraj = np.zeros((nStepsForSim, 2))
    conTraj    = np.zeros((nStepsForSim, 2))
    targTraj   = np.zeros((nStepsForSim, 2))

    ttt        = list()
    trialStart = list()

    # Define kalman & cursor state vectors.
    currControl = np.zeros((2, 1))
    currTarg    = np.random.rand(2,1) - 0.5
    xK          = np.zeros((A_aug.shape[0], 1))
    xC          = np.zeros((A_aug.shape[0], 1))

    for t in range(nStepsForSim):
        #first get the user's control vector
        xK = A_aug.dot(xK) + B_aug.dot(currControl) + L_kalman.dot(C_aug.dot(xC) - C_aug.dot(xK))
        
        posErr      = currTarg - xK[:2, :]
        targDist    = np.linalg.norm(posErr)
        distWeight  = np.interp(targDist, fTarg[:, 0], fTarg[:, 1]) 

        tmp         = (1 / targDist) * distWeight
        currControl = tmp * posErr

        #simulate neural activity tuned to this control vector
        simAct  = neuralTuning[:,1:].dot(currControl) 
        for j in range(len(simAct)):
            simAct[j, 0] = simAct[j, 0] + (np.random.normal() * 0.3) + neuralTuning[j, 0]

        #decode the neural activity with the decoder matrix D
        rawDecVec = np.dot(D[1:,:].T, simAct) 
        for j in range(2):
            rawDecVec[j, 0] = rawDecVec[j, 0] + D[0, j]
            
        #rawDecVec = rawDecVec.T

        #update the cursor state according to the smoothing dynamics
        xC = A_aug.dot(xC) + B_aug.dot(rawDecVec)
        
        #the rest below is target acquisition & trial time out logic
        newTarg = False

        #trial time out
        trialCounter = trialCounter + 1
        if trialCounter > maxSteps:
            newTarg = True
        
        #target acquisition
        trueDist = np.linalg.norm(currTarg - xC[:2])
        if trueDist < targRad:
            holdCounter = holdCounter + 1
        else:
            holdCounter = 0

        if holdCounter > nHoldSteps:
            newTarg = True

        #new target
        if newTarg:
            ttt.append(trialCounter/100)
            trialStart.append(t)

            holdCounter  = 0
            trialCounter = 0
            currTarg     = np.random.rand(2,1) - 0.5

        #finally, save this time step of data
        posTraj[t,:]    = xC[:2, 0]
        posErr_est[t,:] = posErr[:, 0]
        velTraj[t,:]    = xC[2:4, 0]
        rawDecTraj[t,:] = rawDecVec[:, 0]
        conTraj[t,:]    = currControl[:, 0]
        targTraj[t,:]   = currTarg[:, 0]
        neuralTraj[t,:] = simAct[:, 0]
    
    return posTraj, posErr_est, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt

    
    
def simulateBCIFitts_numba(initialTuning, D,alpha, beta, nDelaySteps, delT, nSimSteps, return_PosErr = False):
    '''Define the internal state estimator of the BCI user (which we simulate here with a Kalman filter). 
       Inputs are: 
    
            alpha (float)           - cursor decoder (time) smoothing
            beta (float)            - cursor decoder gain
            nDelaySteps (int)       - delay from visual feedback
            delT (float)            - amount of time per step (ms)
            return_PosErr (Bool)    - if True, return distance to internal target estimate'''
    
    mats     = getBCIFittsKF_py(alpha, beta, nDelaySteps, delT)
    
    posTraj, posErr_est, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt = simulate_numba(*mats, initialTuning, D,alpha, beta, nDelaySteps, delT, nSimSteps)
    
    if return_PosErr:
        return posTraj, posErr_est, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt
    else:
        return posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt
    