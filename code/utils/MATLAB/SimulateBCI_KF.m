function [L_kalman, A_aug, B_aug, C_aug, D_aug, G_aug, H_aug] = SimulateBCI_KF(alpha, beta, nDelaySteps, delT)

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