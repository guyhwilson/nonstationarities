 function [dist_mats, ttt] = generateDistortionMatrices_parallel(neural_list, D, alpha, beta, nDelaySteps, delT, nStepsForSim)

n_trls           = length(neural_list);
[dist_mats, ttt] = deal(cell(n_trls, 1));

parfor i = 1:n_trls
    newTuning = neural_list{i};
    
    [posTraj, velTraj, rawDecTraj, ~, targTraj, neuralTraj, ~, ~] = simulateBCIFitts(newTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim);
    
    PosErr       = targTraj- posTraj;
    D_new        = [ones(length(neuralTraj),1), neuralTraj] \ PosErr;
    decVec       = [ones(size(neuralTraj,1),1), neuralTraj] * D_new;
    
    TargDist     = sqrt(sum(PosErr.^2,2));
    TargDir      = PosErr ./ TargDist;
    farIdx       = find(TargDist>0.4);
    projVec      = sum(decVec(farIdx,:) .* TargDir(farIdx,:),2);
    D_new        = D_new / mean(projVec);
    
    [~, ~, ~, ~, ~, ~, ~, ttt{i}] = simulateBCIFitts(newTuning, D_new, alpha, beta, nDelaySteps, delT, nStepsForSim);
    
    % Calculate distortion matrix: 
    dec          = D(2:end, :);
    enc          = newTuning(:, 2:end);
    dist_mats{i} = (dec' * enc) ./ (vecnorm(dec)' * vecnorm(enc));
        
end

end

