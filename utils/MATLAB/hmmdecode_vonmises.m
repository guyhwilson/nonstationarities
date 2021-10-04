function [pStates, pSeq] = hmmdecode_vonmises(rawDecodeVec, stateTransitions, targLocs, cursorPos, pStateStart, vmKappa, vmAdjust_inflection, vmAdjust_exp)
numStates = size(stateTransitions,1);

% add extra symbols to start to make algorithm cleaner at f0 and b0
L = size(rawDecodeVec,1)+1;

% introduce scaling factors for stability
fs = zeros(numStates,L);
fs(:,1) = pStateStart;

s = zeros(1,L);
s(1) = 1;

for count = 2:L
    %1. compute distance from the cursor to each target, and expected angle
    %for that target
    tDists = sqrt(sum((targLocs - cursorPos(count-1,:)).^2,2));
    normPosErr = (targLocs - cursorPos(count-1,:))./tDists;
    expectedAngle = atan2(normPosErr(:,2), normPosErr(:,1));
    
    %2. compute expected precision based on the base kappa and distance to
    %target (very close distances -> very large dispersion in expected
    %angles)
    vmKappa_adjusted = vmKappa * 1./(1+exp(-(tDists- vmAdjust_inflection ) * vmAdjust_exp));
    
    %3. compute VM probability densities
    observedAngle = atan2(rawDecodeVec(count-1,2), rawDecodeVec(count-1,1));
    vmProbLog = exp(vmKappa_adjusted.*cos(observedAngle - expectedAngle))./(2*pi*besseli(0,vmKappa_adjusted));
    
    fs(:,count) = vmProbLog .* (stateTransitions'*fs(:,count-1));

    s(count) =  sum(fs(:,count));
    fs(:,count) =  fs(:,count)./s(count);
end

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

