import numpy as np 

import sys
sys.path.append('utils/MATLAB/')
from simulation_matlab import simulateBCI_KF

def getBCIFittsKF(neuralTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim):
    '''
    we define the internal state estimator of the BCI user (which we simulate
    here with a Kalman filter).
    
    We begin by defining the decoder dynamics, which are a simple first-order
    smoother + integrator (damped point mass dynamics). '''
   
    A = np.array([[1, 0, delT, 0],
        [0, 1, 0, delT],
        [0, 0, alpha, 0],
        [0, 0, 0, alpha]])

    B = np.array([[0, 0], [0, 0],
                [beta*(1-alpha), 0],
                [0, beta*(1-alpha)]])
    
    C = np.eye(4)

    # Now we augment the state of this plant to define a feedback delay.
    nStates = A.shape[0]
    A_aug   = np.zeros((nStates*(1+nDelaySteps), nStates*(1+nDelaySteps)))

    A_aug[:nStates, :nStates] = A

    for d in range(nDelaySteps):
        rowIdx = np.arange(nStates*d + 1, nStates*(d+1))
        colIdx = rowIdx - nStates

        for x in range(len(rowIdx)):
            A_aug[rowIdx[x],colIdx[x]] = 1


    B_aug              = np.zeros((A.shape[0], B.shape[1]))
    B_aug[:nStates, :] = B

    C_aug                      = np.zeros(C.shape[0], A_aug.shape[0])
    C_aug[:,(-1 * nStates+1):] = C

    D_aug = np.zeros(C_aug.shape[0], B.shape[1])

    G_aug = np.zeros((A_aug.shape[0], nStates))
    H_aug = np.zeros((D_aug.shape[0], nStates))

    G_aug[:nStates, :nStates] = np.eye(nStates)

    #Finally, solve for a kalman estimator of this plant that can estimate its
    #current state from delayed feedback only. TODO: IMPLEMENT
    #sysDelay = ss(A_aug, [B_aug G_aug], C_aug, [D_aug H_aug], delT);


    QN = np.eye(4)
    RN = np.eye(4)
    [KEST, L_kalman] = kalman(sysDelay, QN, RN);
    #-------------------------------
    '''
    %%
    %The following commented out code can be uncommented in debug mode to
    %test out the Kalman filter and make sure it works. 

    %x0 = zeros(size(A_aug,1),1);
    %x0(3) = 1;

    %currState = x0;
    %currKalmanState = zeros(size(KEST.A,1),1);

    %traj = [];
    %kalmanTraj = [];
    %for t=1:100
    %   traj = [traj; currState'];
    %   kalmanTraj = [kalmanTraj; currKalmanState'];

    %   currState = A_aug*currState;
    %   currKalmanState = A_aug*currKalmanState + L_kalman*(C_aug*currState - C_aug*currKalmanState);
    %end

    %figure
    %hold on;
    %plot(traj(:,1:4));
    %plot(kalmanTraj(:,1:4),'--','LineWidth',2);
    '''              


                      
def simulateBCIFitts(neuralTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim, return_PosErr = False):
    ''' Simulate BCI Fitts task. Inputs are: 
    
    
    '''
    
    mats     = simulateBCI_KF(alpha, beta, nDelaySteps, delT)
    L_kalman = mats[0]
    A_aug    = mats[1]
    B_aug    = mats[2]
    C_aug    = mats[3]

    '''
    Here we define the fTarg function for the simulated user's control policy, which
    is basically just a saturating nonlinearity (see the paper
    'Principled BCI decoder design and parameter selection using a
    feedback control model' for more details).
    '''
    
    fTarg = np.array([[0.1373,    0.0000],
                    [10.9670,   150.1874],
                    [41.0618,   180.8933],
                    [68.8483,  250.1018],
                    [104.7274 , 250.4014],
                    [150.9665 , 260.4379],
                    [204.6272 , 272.2782],
                    [263.8003 , 280.9718],
                    [329.5381 , 309.6743],
                    [399.8577 , 299.3726]])
      
    fTarg[:, 0] /= 400
    fTarg[:, 1] /= 300


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
        

        posErr      = currTarg - xK[:2]
        targDist    = np.linalg.norm(posErr)
        distWeight  = np.interp(targDist, 
                                np.concatenate([[0], fTarg[:,0], [100]]), 
                                np.concatenate([[fTarg[0,1]], fTarg[:,1], [fTarg[-1,1]]])  ) 

        #currControl  =  posErr / targDist # we're ignoring the velocity dampening term
        currControl = distWeight * (posErr/targDist)


        #simulate neural activity tuned to this control vector
        simAct  = neuralTuning[:,1:].dot(currControl) + neuralTuning[:, 0][:, np.newaxis]
        simAct += np.random.normal(size = simAct.shape) * 0.3

        #decode the neural activity with the decoder matrix D
        rawDecVec = simAct.T.dot(D[1:,:]) + D[0,:]
        rawDecVec = rawDecVec.T

        #update the cursor state according to the smoothing dynamics
        xC = A_aug.dot(xC) + B_aug.dot(rawDecVec)
        
        #the rest below is target acquisition & trial time out logic
        newTarg = False

        #trial time out
        trialCounter += 1
        if trialCounter > maxSteps:
            newTarg = True
        
        #target acquisition
        trueDist = np.linalg.norm(currTarg - xC[:2])
        if trueDist < targRad:
            holdCounter += 1
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
        posTraj[t,:]    = xC[:2].squeeze()
        posErr_est[t,:] = posErr.squeeze()
        velTraj[t,:]    = xC[2:4].squeeze()
        rawDecTraj[t,:] = rawDecVec.squeeze()
        conTraj[t,:]    = currControl.squeeze()
        targTraj[t,:]   = currTarg.squeeze()
        neuralTraj[t,:] = simAct.squeeze()
    
    if return_PosErr:
        return posTraj, posErr_est, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt
    else:
        return posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt
    
    

def get_ClickMagnitude(distance, prev):
    '''Drift-diffusion model with decay for click signal. Inputs are:
    
        distance (float) - internal estimate of distance from target
        prev (float)     - previously encoded click magnitude '''
    
        
    alpha    = 0.01
    decay    = -0.005
    noisevar = 0.005
        
    evidence = 1 / (1 + np.exp(1 * (distance - 0.07) * 20)) 
    mag      = max(prev +  (alpha * evidence) + np.random.normal(decay, noisevar), 0)
    
    return mag
    
    
    
    
def simulateBCIFitts_click(cursorTuning, clickTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim, return_PosErr = False):
    ''' Simulate BCI Fitts task with click decoding. Inputs are: 
    
    
    TODO:
     - implement click magnitude fxn
     - project click magnitude onto a click dimension
     - implement a click decoder
     - shift trial time logic to measure correct click 
    '''
    
    mats     = simulateBCI_KF(alpha, beta, nDelaySteps, delT)
    L_kalman = mats[0]
    A_aug    = mats[1]
    B_aug    = mats[2]
    C_aug    = mats[3]

    '''
    Here we define the fTarg function for the simulated user's control policy, which
    is basically just a saturating nonlinearity (see the paper
    'Principled BCI decoder design and parameter selection using a
    feedback control model' for more details).
    '''
    
    fTarg = np.array([[0.1373,    0.0000],
                    [10.9670,   150.1874],
                    [41.0618,   180.8933],
                    [68.8483,  250.1018],
                    [104.7274 , 250.4014],
                    [150.9665 , 260.4379],
                    [204.6272 , 272.2782],
                    [263.8003 , 280.9718],
                    [329.5381 , 309.6743],
                    [399.8577 , 299.3726]])
      
    fTarg[:, 0] /= 400
    fTarg[:, 1] /= 300


    # This is the main simulation loop.
    # These are some task parameters.
    nHoldSteps = 50
    holdCounter = 0
    trialCounter = 0
    maxSteps = 1000
    targRad = 0.075

    # Define matrices to hold the time series data.
    posTraj      = np.zeros((nStepsForSim, 2))
    posErr_est   = np.zeros((nStepsForSim, 2))     # internal estimate of position error
    neuralTraj   = np.zeros((nStepsForSim, cursorTuning.shape[0]))
    velTraj      = np.zeros((nStepsForSim, 2))
    rawDecTraj   = np.zeros((nStepsForSim, 2))
    clickMagTraj = np.zeros((nStepsForSim, 1))   # magnitude of underlying click signal 
    clickTraj    = np.zeros((nStepsForSim, 1))   # decoded click signal 
    conTraj      = np.zeros((nStepsForSim, 2))
    targTraj     = np.zeros((nStepsForSim, 2))

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
        

        posErr      = currTarg - xK[:2]
        targDist    = np.linalg.norm(posErr)
        distWeight  = np.interp(targDist, 
                                np.concatenate([[0], fTarg[:,0], [100]]), 
                                np.concatenate([[fTarg[0,1]], fTarg[:,1], [fTarg[-1,1]]])  ) 

        #currControl  =  posErr / targDist # we're ignoring the velocity dampening term
        currControl = distWeight * (posErr/targDist)
        
        # get click control signal 
        clickMag = get_ClickMagnitude(targDist, clickMagTraj[t-1])


        #simulate neural activity tuned to this control vector and to the click signal
        simAct  = cursorTuning[:,1:].dot(currControl) + cursorTuning[:, 0][:, np.newaxis]
        simAct += np.random.normal(size = simAct.shape) * 0.3
        
        #simAct += (clickMag * clickTuning) + np.random.normal(size = simAct.shape) * 0.3
        
        

        #decode the neural activity with the decoder matrix D
        rawDecVec = simAct.T.dot(D[1:,:]) + D[0,:]
        rawDecVec = rawDecVec.T

        #update the cursor state according to the smoothing dynamics
        xC = A_aug.dot(xC) + B_aug.dot(rawDecVec)
        
        #the rest below is target acquisition & trial time out logic
        newTarg = False

        #trial time out
        trialCounter += 1
        if trialCounter > maxSteps:
            newTarg = True
        
        #target acquisition
        trueDist = np.linalg.norm(currTarg - xC[:2])
        if trueDist < targRad:
            holdCounter += 1
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
        posTraj[t,:]      = xC[:2].squeeze()
        posErr_est[t,:]   = posErr.squeeze()
        velTraj[t,:]      = xC[2:4].squeeze()
        rawDecTraj[t,:]   = rawDecVec.squeeze()
        clickMagTraj[t,:] = clickMag
        conTraj[t,:]      = currControl.squeeze()
        targTraj[t,:]     = currTarg.squeeze()
        neuralTraj[t,:]   = simAct.squeeze()
    
    return posTraj, posErr_est, velTraj, rawDecTraj, clickMagTraj, clickTraj, conTraj, targTraj, neuralTraj, trialStart, ttt
 
    
    
    
    







