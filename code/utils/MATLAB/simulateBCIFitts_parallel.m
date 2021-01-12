function [ttt] = simulateBCIFitts_parallel(neural_list, D, alpha, beta, nDelaySteps, delT, nStepsForSim)
   
	
    n_trls = length(neural_list);
    [posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt] = deal(cell(n_trls, 1));
    
    parfor i = 1:n_trls
        neuralTuning = neural_list{i};
        
        %[posTraj{i}, velTraj{i}, rawDecTraj{i}, conTraj{i}, targTraj{i}, neuralTraj{i}, trialStart{i}, ttt{i}] = simulateBCIFitts(neuralTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim);
        simulateBCIFitts(neuralTuning, D, alpha, beta, nDelaySteps, delT, nStepsForSim);

        
    end
    
end