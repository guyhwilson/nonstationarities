function [posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt] = simulateBCIFitts(neuralTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim)
    %%
    %First we define the internal state estimator of the BCI user (which we simulate
    %here with a Kalman filter).
    
    %We begin by defining the decoder dynamics, which are a simple first-order
    %smoother + integrator (damped point mass dynamics). See 'help ss' for
    %definitions of these variables.
    A = [1 0 delT 0;
        0 1 0 delT;
        0 0 alpha 0;
        0 0 0 alpha];
    
    B = [0 0;
        0 0;
        beta*(1-alpha) 0;
        0 beta*(1-alpha)];
    
    C = eye(4);

    %Now we augment the state of this plant to define a feedback delay.
    nStates = size(A,1);
    A_aug = zeros(nStates*(1+nDelaySteps), nStates*(1+nDelaySteps));
    A_aug(1:nStates, 1:nStates) = A;

    for d=1:nDelaySteps
        rowIdx = (nStates*d + 1):(nStates*(d+1));
        colIdx = rowIdx - nStates;

        for x=1:length(rowIdx)
            A_aug(rowIdx(x),colIdx(x)) = 1; 
        end
    end

    B_aug = zeros(size(A_aug,1), size(B, 2));
    B_aug(1:nStates, :) = B;

    C_aug = zeros(size(C, 1), size(A_aug,1));
    C_aug(:,(end-nStates+1):end) = C;

    D_aug = zeros(size(C_aug,1), size(B,2));

    G_aug = zeros(size(A_aug,1), nStates);
    H_aug = zeros(size(D_aug,1), nStates);

    G_aug(1:nStates, 1:nStates) = eye(nStates);
    
    %Finally, solve for a kalman estimator of this plant that can estimate its
    %current state from delayed feedback only.
    sysDelay = ss(A_aug, [B_aug G_aug], C_aug, [D_aug H_aug], delT);

    QN = eye(4);
    RN = eye(4);
    [KEST, L_kalman] = kalman(sysDelay, QN, RN);
    
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
    
    %%
    %Here we define the fTarg function for the simulated user's control policy, which
    %is basically just a saturating nonlinearity (see the paper
    %'Principled BCI decoder design and parameter selection using a
    %feedback control model' for more details).
    
    fTarg = [0.1373    0.0000
           10.9670   150.1874
           41.0618   180.8933
           68.8483  250.1018
          104.7274  250.4014
          150.9665  260.4379
          204.6272  272.2782
          263.8003  280.9718
          329.5381  309.6743
          399.8577  299.3726];
      
    fTarg(:,1) = fTarg(:,1) / 400;
    fTarg(:,2) = fTarg(:,2) / 300;
    
    %%
    %This is the main simulation loop.
    
    %These are some task parameters.
    nHoldSteps = 50;
    holdCounter = 0;
    trialCounter = 0;
    maxSteps = 1000;
    targRad = 0.075;
    
    %Define matrices to hold the time series data.
    posTraj = zeros(nStepsForSim, 2);
    neuralTraj = zeros(nStepsForSim, size(neuralTuning,1));
    velTraj = zeros(nStepsForSim, 2);
    rawDecTraj = zeros(nStepsForSim, 2);
    conTraj = zeros(nStepsForSim, 2);
    targTraj = zeros(nStepsForSim, 2);
    
    ttt = [];
    trialStart = [];
    
    %Define kalman & cursor state vectors.
    currControl = [0; 0];
    currTarg = (rand(2,1)-0.5);
    xK = zeros(size(A_aug,1),1);
    xC = zeros(size(A_aug,1),1); 
    
    for t=1:nStepsForSim
        %first get the user's control vector
        xK = A_aug*xK + B_aug*currControl + L_kalman*(C_aug*xC - C_aug*xK);
        
        posErr      = currTarg - xK(1:2);
        targDist    = sqrt(sum(posErr.^2));
        distWeight  = interp1([0; fTarg(:,1); 100], [fTarg(1,2); fTarg(:,2); fTarg(end,2)], targDist);   % we're ignoring the velocity dampening term
        currControl =  (posErr/targDist);
        %currControl = distWeight * (posErr/targDist);

        
        %simulate neural activity tuned to this control vector
        simAct = neuralTuning(:,2:3)*currControl + neuralTuning(:,1);
        simAct = simAct + randn(size(simAct))*0.3;
        
        %decode the neural activity with the decoder matrix D
        rawDecVec = simAct'*D(2:end,:) + D(1,:);
        rawDecVec = rawDecVec';
        
        %update the cursor state according to the smoothing dynamics
        xC = A_aug*xC + B_aug*rawDecVec;
        
        %the rest below is target acquisition & trial time out logic
        newTarg = false;
        
        %trial time out
        trialCounter = trialCounter + 1;
        if trialCounter > maxSteps
            newTarg = true;
        end
        
        %target acquisition
        trueDist = sqrt(sum((currTarg - xC(1:2)).^2));
        if trueDist < targRad
            holdCounter = holdCounter + 1;
        else
            holdCounter = 0;
        end
        if holdCounter > nHoldSteps
            newTarg = true;
        end
        
        %new target
        if newTarg
            ttt = [ttt; trialCounter/100];
            trialStart = [trialStart; t];
            
            holdCounter = 0;
            trialCounter = 0;
            currTarg = (rand(2,1)-0.5);
        end
        
        %finally, save this time step of data
        posTraj(t,:) = xC(1:2);
        velTraj(t,:) = xC(3:4);
        rawDecTraj(t,:) = rawDecVec;
        conTraj(t,:) = currControl;
        targTraj(t,:) = currTarg;
        neuralTraj(t,:) = simAct;
    end
    
end