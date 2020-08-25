import numpy as np
import math

def hmmdecode_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, pStateStart, vmKappa):
  
  numStates = stateTransitions.shape[0]

  # add extra symbols to start to make algorithm cleaner at f0 and b0
  L = rawDecodeVec.shape[0] +1

  # introduce scaling factors for stability
  fs      = np.zeros((numStates,L))
  fs[:,0] = pStateStart

  s       = np.zeros((L,))
  s[0]    = 1;

  for count in range(1, L):
    #1. compute distance from the cursor to each target, and expected angle for that target:
    tDists        = np.sqrt(np.sum((targLocs - cursorPos[count-1,:])**2, axis = 1))
    normPosErr    = np.divide((targLocs - cursorPos[count-1,:]), tDists)
    expectedAngle = np.atan2(normPosErr[:,1], normPosErr[:,0])
    
    #2. compute expected precision based on the base kappa and distance to
    #   target (very close distances -> very large dispersion in expected angles):
    
    # CONTINUE EDITING BELOW
    vmKappa_adjusted = vmKappa * 1./(1+exp(-(tDists-0.1)*20));
    
    %3. compute VM probability densities
    observedAngle = atan2(rawDecodeVec(count-1,2), rawDecodeVec(count-1,1));
    vmProbLog = exp(vmKappa_adjusted.*cos(observedAngle - expectedAngle))./(2*pi*besseli(0,vmKappa_adjusted));
    
    fs(:,count) = vmProbLog .* (stateTransitions'*fs(:,count-1));

    s(count) =  sum(fs(:,count));
    fs(:,count) =  fs(:,count)./s(count);


bs = ones(numStates,L);
for count = L-1:-1:1
    %1. compute distance from the cursor to each target, and expected angle
    %for that target
    tDists = sqrt(sum((targLocs - cursorPos(count,:)).^2,2));
    normPosErr = (targLocs - cursorPos(count,:))./tDists;
    expectedAngle = atan2(normPosErr(:,2), normPosErr(:,1));
    
    %2. compute expected precision based on the base kappa and distance to
    %target (very close distances -> very large dispersion in expected
    %angles)
    vmKappa_adjusted = vmKappa * 1./(1+exp(-(tDists-0.1)*20));
    
    %3. compute VM probability densities
    observedAngle = atan2(rawDecodeVec(count,2), rawDecodeVec(count,1));
    vmProbLog = exp(vmKappa_adjusted.*cos(observedAngle - expectedAngle))./(2*pi*besseli(0,vmKappa_adjusted));
    
    probWeightBS = bs(:,count+1).*vmProbLog;
    tmp = stateTransitions * probWeightBS;
    bs(:,count) = tmp*(1/s(count+1));
end

pSeq = sum(log(s));
pStates = fs.*bs;

% get rid of the column that we stuck in to deal with the f0 and b0 
pStates(:,1) = [];
                                
                                return pStates, pSeq

                                
                                
function [currentState, logP] = hmmviterbi_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, pStateStart, vmKappa)

numStates = size(stateTransitions,1);
L = size(rawDecodeVec,1);
currentState = zeros(1,L);
pTR = zeros(numStates,L);

% work in log space to avoid numerical issues
logTR = log(stateTransitions);
v = log(pStateStart);
vOld = v;

% loop through the model
for count = 1:L
    %von mises emissions probabilities
    
    %1. compute distance from the cursor to each target, and expected angle
    %for that target
    tDists = sqrt(sum((targLocs - cursorPos(count,:)).^2,2));
    normPosErr = (targLocs - cursorPos(count,:))./tDists;
    expectedAngle = atan2(normPosErr(:,2), normPosErr(:,1));
    
    %2. compute expected precision based on the base kappa and distance to
    %target (very close distances -> very large dispersion in expected
    %angles)
    vmKappa_adjusted = vmKappa * 1./(1+exp(-(tDists-0.1)*20));
    
    %3. compute VM probability densities
    observedAngle = atan2(rawDecodeVec(count,2), rawDecodeVec(count,1));
    vmProbLog = (vmKappa_adjusted.*cos(observedAngle - expectedAngle)) - log(2*pi*besseli(0,vmKappa_adjusted));
    
    tmpV = vOld + logTR;
    [maxVal, maxIdx] = max(tmpV);
    
    pTR(:,count) = maxIdx;
    v = vmProbLog + maxVal';
    vOld = v;
end

% decide which of the final states is post probable
[logP, finalState] = max(v);

% Now back trace through the model
currentState(L) = finalState;
for count = L-1:-1:1
    currentState(count) = pTR(currentState(count+1),count+1);
    if currentState(count) == 0
        error(message('stats:hmmviterbi:ZeroTransitionProbability', currentState( count + 1 )));
    end
end

