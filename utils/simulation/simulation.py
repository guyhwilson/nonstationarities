import numpy as np 
from numba import jit, types
import numba
from scipy.linalg import solve_discrete_are
import sys


def getBCIFittsKF(alpha, beta, nDelaySteps, delT):
    '''Define the internal state estimator of the BCI user (which we simulate here with a Kalman filter). 
       Inputs are: 
    
            alpha (float)           - cursor decoder (time) smoothing
            beta (float)            - cursor decoder gain
            nDelaySteps (int)       - delay from visual feedback
            delT (float)            - amount of time per step (ms)
    
    begin by defining the decoder dynamics, which are a simple first-order
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
    nObs    = C.shape[0]
    A_aug   = np.zeros((nStates*(1+nDelaySteps), nStates*(1+nDelaySteps)))
    A_aug[:nStates, :nStates] = A

    for d in range(nDelaySteps):
        rowIdx = np.arange(nStates*(d + 1) , nStates*(d+2))
        colIdx = rowIdx - nStates 
        for x in range(len(rowIdx)):
            A_aug[rowIdx[x],colIdx[x]] = 1

    B_aug                    = np.zeros((A_aug.shape[0], B.shape[1]))
    B_aug[:nStates, :]       = B
    C_aug                    = np.zeros((C.shape[0], A_aug.shape[0]))
    C_aug[:,(-1 * nStates):] = C

    D_aug = np.zeros((C_aug.shape[0], B.shape[1]))
    G_aug = np.zeros((A_aug.shape[0], nStates))
    H_aug = np.zeros((D_aug.shape[0], nStates))

    G_aug[:nStates, :nStates] = np.eye(nStates)
    
    '''Finally, solve for a kalman estimator of this plant that can estimate its
    current state from delayed feedback only. Model is:   
    
                  x_t+1 = A_aug x_t + B_aug u_t + G_aug w_t,   G_aug w_t ~ N(0, G_aug Q G_aug^T)
                  y_y   = C_aug x_t + D_aug u_t + H_aug u_t + v_t,   v_t ~ N(0, R)
                  
    we assume that N = Cov(v_t, w_t) = 0, and that D_aug = H_aug = 0  '''
    Q        = np.eye(4)
    R        = np.eye(4)
    L_kalman = np.zeros((nObs, nStates))
    
    # solve for steady-state latent measurement noise covariance: 
    P        = solve_discrete_are(a = A_aug.T, b = C_aug.T, q = G_aug.dot(Q).dot(G_aug.T), r = R)
    
    # solve discrete riccati equation: 
    t1       = P.dot(C_aug.T) 
    t2       = np.linalg.inv(C_aug.dot(P).dot(C_aug.T) + R)
    L_kalman = t1.dot(t2)
    
    return L_kalman, A_aug, B_aug, C_aug, D_aug, G_aug, H_aug




@jit(nopython=True) 
def getNeuralTuning(currControl, decode_params):
    '''Simulate linear tuning to control vector signal.'''
    
    simAct  = decode_params['neuralTuning'][:,1:].dot(currControl) 
    for j in range(len(simAct)):
        simAct[j, 0] = simAct[j, 0] + (np.random.normal() * 0.3) +  decode_params['neuralTuning'][j, 0]
    
    return simAct


@jit(nopython = True)
def getDecodedControl(simAct, decode_params):
    '''Decode control signal from neural signals.'''
    
    rawDecVec = np.dot(decode_params['D'][1:,:].T, simAct) 
    for j in range(rawDecVec.shape[0]):
        rawDecVec[j, 0] = rawDecVec[j, 0] + decode_params['D'][0, j]

    return rawDecVec

@jit(nopython = True)
def applyWallBound(state_vector, bound = 0.5):
    '''Stop cursor from shooting out to infinity by having walls'''
    
    for i in range(2):
        state_vector[i, 0] = min(np.abs(state_vector[i, 0]), bound) * np.sign(state_vector[i, 0])
    
    return state_vector
    
    



@jit(nopython=True) 
def simulateFitts_Numba(L_kalman, A_aug, B_aug, C_aug, d_aug, g_aug, h_aug, decode_params, params):
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
    
    
    
    nDelaySteps  = int(params['nDelaySteps'])
    delT         = params['delT']
    nStepsForSim = int(params['nSimSteps'])
    
    
    neuralTuning = decode_params['neuralTuning']
    D            = decode_params['D']
    alpha        = decode_params['alpha'][:, 0]
    beta         = decode_params['beta'][:, 0]


    # This is the main simulation loop.
    # These are some task parameters.
    nHoldSteps = 50
    holdCounter  = 0
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
        
        #xK = applyWallBound(xK, bound = 0.5)
        
        posErr      = currTarg - xK[:2, :]
        targDist    = np.linalg.norm(posErr)
        distWeight  = np.interp(targDist, fTarg[:, 0], fTarg[:, 1]) 
        
        if targDist > 1e-20:
            tmp = (1 / targDist) * distWeight
        else:
            tmp = 0
        currControl = tmp * posErr

        #simulate neural activity tuned to this control vector
        simAct  = getNeuralTuning(currControl, decode_params)

        #decode the neural activity with the decoder matrix D
        rawDecVec = getDecodedControl(simAct, decode_params)
       

        #update the cursor state according to the smoothing dynamics
        xC = A_aug.dot(xC) + B_aug.dot(rawDecVec)
        #xC = applyWallBound(xC, bound = 0.5)
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


    
def simulateBCIFitts(cfg):
    '''Define the internal state estimator of the BCI user (which we simulate here with a Kalman filter). 
       Input <param> is a dictionary with key-value pairs:
       
            neuralTuning (2D array) - channels x 3 array of means, x, and y-velocity encoding dimensions
            D (2D array)            - (channnels + 1) x 2 array for cursor decoder
            alpha (float)           - cursor decoder (time) smoothing
            beta (float)            - cursor decoder gain
            nDelaySteps (int)       - delay from visual feedback
            delT (float)            - amount of time per step (ms)
         '''
    
    mats = getBCIFittsKF(cfg['alpha'], cfg['beta'], cfg['nDelaySteps'], cfg['delT'])
    
    params = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    
    params['delT']        = float(cfg['delT'])
    params['nSimSteps']   = float(cfg['nSimSteps'])
    params['nDelaySteps'] = float(cfg['nDelaySteps'])
    
    decode_params = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64[:, :])
    
    decode_params['neuralTuning'] = cfg['neuralTuning']
    decode_params['D']            = cfg['D']
    decode_params['alpha']        = np.asarray([cfg['alpha']], dtype = np.float64)[:, None]
    decode_params['beta']         = np.asarray([cfg['beta']], dtype = np.float64)[:, None]

    out   = simulateFitts_Numba(*mats, decode_params, params)
    keys  = ['posTraj', 'posErr_hat', 'velTraj', 'rawDecTraj', 'conTraj', 
            'targTraj', 'neuralTraj', 'trialStart', 'ttt']
    
    results = dict(zip(keys, out))
    
    return results
    
  
    

def get_ClickMagnitude(distance, prev):
    '''Drift-diffusion model with decay for click signal. Inputs are:
    
        distance (float) - internal estimate of distance from target
        prev (float)     - previously encoded click magnitude '''
    
        
    alpha    = 0.1
    decay    = -0.05
    noisevar = 0.05
    evidence = 1 / (1 + np.exp(1 * (distance - 0.07) * 20)) 
    mag      = max(prev +  (alpha * evidence) + np.random.normal(decay, noisevar), 0) 
    mag      = min(mag, 1)
    
    return mag
    
    
    
    
def simulateBCIFitts_click(cursorTuning, clickTuning, D, D_click, alpha, beta, nDelaySteps, delT, nStepsForSim, return_PosErr = False, ClickLogic = True, ConsecutiveClicksNeeded = 1):
    ''' Simulate BCI Fitts task with click decoding. Inputs are: 
    
            cursorTuning (2D array)       - channels x 3 array holding baseline FR, X-velocity tuning, and Y-velocity tuning
            clickTuning (2D array)        - channels x 1 array containing click signal tuning
            D (2D array)                  - (channnels x 1) x 2 array for cursor decoder
            D_click (object)              - has method predict_proba() that takes in instantaneous neural signal and outputs click probability
            alpha (float)                 - cursor decoder (time) smoothing
            beta (float)                  - cursor decoder gain
            nDelaySteps (int)             - delay from visual feedback
            delT (float)                  - amount of time per step (ms)
            ClickLogic (bool)             - if True, use click decoder to select targets; else use dwell logic
            ConsecutiveClicksNeeded (int) - wait for this many consecutive clicks before issuing a click
             ''' 
    
    L_kalman, A_aug, B_aug, C_aug = getBCIFittsKF(alpha, beta, nDelaySteps, delT)[:4]    

    # define the fTarg function for the simulated user's control policy, which is basically just a saturating nonlinearity
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
    nHoldSteps        = 50  # for dwell - number of timesteps to wait before selecting a target
    holdCounter       = 0   # for dwell - stores the number of timesteps waited on thus far for target selection
    trialCounter      = 0
    clickCooldown     = False 
    consecutiveClicks = 0
    cooldownCounter   = True
    maxSteps          = 1000
    targRad           = 0.075

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

        currControl = distWeight * (posErr/targDist)                   # we're ignoring the velocity dampening term
        clickMag    = get_ClickMagnitude(targDist, clickMagTraj[t-1])  # get click control signal 


        #simulate neural activity tuned to this control vector and to the click signal
        simAct  = cursorTuning[:,1:].dot(currControl) + cursorTuning[:, 0][:, np.newaxis]
        simAct += np.random.normal(size = simAct.shape) * 0.3
        
        simAct         += clickMag * clickTuning
        
        
        #decode the neural activity with the decoder matrix D and click decoder
        rawDecVec = simAct.T.dot(D[1:,:]) + D[0,:]
        rawDecVec = rawDecVec.T
        
        if ClickLogic == False and D_click is None:
            rawDecClick = 0
        else:
            # small hack: we integrate the click decoder's predictions over a 2-timebin window to reduce noise
            clickwin    = np.hstack([simAct, neuralTraj[t-1, :][:, np.newaxis]])
            rawDecClick = np.mean(D_click[0].predict_proba(clickwin.T)[:, 1])
            rawDecClick = float(rawDecClick > D_click[1])
            
            consecutiveClicks += rawDecClick # if click decoded, count stays same; else resets to 0
            consecutiveClicks += rawDecClick # if click decoded, count goes up by 1; else remains at 0 
        
        
        #update the cursor state according to the smoothing dynamics
        xC = A_aug.dot(xC) + B_aug.dot(rawDecVec)
        
        #the rest below is target acquisition & trial time out logic
        newTarg = False

        #trial time out
        trialCounter += 1
        if trialCounter > maxSteps:
            newTarg = True
        
        #---target acquisition------
        trueDist = np.linalg.norm(currTarg - xC[:2])
        
        # click-based target selection:
        if ClickLogic:
            if clickCooldown:
                cooldownCounter += 1
                rawDecClick      = 0
                if cooldownCounter > 10:  # cooldown period for 10 * delT milliseconds (150 ms)
                    clickCooldown   = False
                    cooldownCounter = 0

            
            if trueDist < targRad:
                if rawDecClick == 1 and clickCooldown == False and consecutiveClicks >= ConsecutiveClicksNeeded:
                    newTarg       = True
                    clickCooldown = True
                    
        # dwell time based target selection      
        else:
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
        clickTraj[t,:]    = rawDecClick
        conTraj[t,:]      = currControl.squeeze()
        targTraj[t,:]     = currTarg.squeeze()
        neuralTraj[t,:]   = simAct.squeeze()
        
    if return_PosErr:
        return posTraj, posErr_est, velTraj, rawDecTraj, clickMagTraj, clickTraj, conTraj, targTraj, neuralTraj, np.asarray(trialStart), np.asarray(ttt)
    else:
        return posTraj, velTraj, rawDecTraj, clickMagTraj, clickTraj, conTraj, targTraj, neuralTraj, np.asarray(trialStart), np.asarray(ttt)
            
