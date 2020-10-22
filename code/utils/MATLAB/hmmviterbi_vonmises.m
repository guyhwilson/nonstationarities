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

% decide which of the final states is most probable
[logP, finalState] = max(v);

% Now back trace through the model
currentState(L) = finalState;
for count = L-1:-1:1
    currentState(count) = pTR(currentState(count+1),count+1);
    if currentState(count) == 0
        error(message('stats:hmmviterbi:ZeroTransitionProbability', currentState( count + 1 )));
    end
end


