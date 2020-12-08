%%
%for a reproducible result
rng(1);

%%
%Define an initial decoder and initial neural tuning properties (mean
%firing rates and preferred directions).
nUnits = 100;

%The first column of initialTuning is the means, and the second two columns
%are the preferred directions. We make sure that the last two columns are
%orthogonal (uniformly distributed PDs) and that the tuning strength is set
%to 1 (norm of the column is 1). 
initialTuning = randn(nUnits, 3); 
[initialTuning(:,2:3), R] = qr(initialTuning(:,2:3), 0);

%The following is a really simple way to build a linear decoder based on
%the above tuning properties. First we define some velocity vectors
%(calVelocity), then simulate neural tuning to those vectors (calNeural),
%and finally use ordinary least squares regression to find a decoder (D) that
%predicts calVelocity from calNeural.
nTrainingSteps = 10000;
calVelocity = randn(nTrainingSteps, 2);
calNeural = calVelocity*initialTuning(:,2:3)' + initialTuning(:,1)';
calNeural = calNeural + randn(size(calNeural))*0.3;

D = [ones(size(calNeural,1),1), calNeural] \ calVelocity;

%Normalize the gain of this decoder so that it will output vectors with a
%magnitude of 1 when the encoded velocity has a magnitude of 1. 
D(:,1) = D(:,1) / norm(D(2:end,1)) / norm(initialTuning(:,2));
D(:,2) = D(:,2) / norm(D(2:end,2)) / norm(initialTuning(:,3));

%Here we define the amount of exponential smoothing used in the decoder (alpha). Values
%between 0.9 and 0.96 are pretty reasonable. See the paper 
%'A comparison of intention estimation methods for decoder calibration in
%intracortical brain–computer interfaces' for an explanation of how velocity Kalman
%filters can be parameterized with a smoothing parameter (alpha), gain
%parameter (beta, see next section below) and decoding matrix (D). 
alpha = 0.94;

%define the time step (10 ms)
delT = 0.01;

%define the simulated user's visual feedback delay (200 ms)
nDelaySteps = 20;

%%
%Do a quick sweep of cursor gains to find the optimal one for this task. This is
%really important so that any new recalibration algorithm doesn't improve
%performance simply by coincidence, via randomly changing the gain to some better
%value.

%The task we are simulating is a fitts style task where targets randomly
%appear within a box centered at the origin. To acquire the target, the
%user must hold the cursor on top of the target for half a second. The
%performance metric of interest is the average total trial time (TTT), or
%the average amount of time it takes to reach to and fully hold on a target. 

%The 'simulateBCIFitts' does all the work of simulating the BCI user, the
%neural activity, and the decoder. It returns time series data you can use
%to see the cursor trajectory and target locations (posTraj & velTraj are
%the cursor positions and velocities, rawDecTraj is the raw decoded
%velocity vectors before they are smoothed, conTraj is the user's internal
%control vector, targTraj is a time series of target locations, neuralTraj
%is a time series of neural activity, trialStart contains the time step on
%which each trial started, and ttt has the trial time (in seconds) for each
%trial. 

possibleGain = linspace(0.5,2.5,10);
meanTTT = zeros(length(possibleGain),1);
for g=1:length(possibleGain)
    disp([num2str(g) ' / ' num2str(length(possibleGain))]);
    nSimSteps = 50000;
    [posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt] = ...
        simulateBCIFitts(initialTuning, D, alpha, possibleGain(g), nDelaySteps, delT, nSimSteps);
    meanTTT(g) = mean(ttt);
end

figure
plot(possibleGain, meanTTT, '-o');
xlabel('Gain');
ylabel('Mean Trial Time (s)');

[~,minIdx] = min(meanTTT);
beta = possibleGain(minIdx);
disp(['Using gain value beta = ' num2str(beta)]);

%%
%Simulate BCI performance with matched neural tuning and decoder, and an optimized gain
nSimSteps = 100000;
[posTraj, velTraj, rawDecTraj, conTraj, targTraj, neuralTraj, trialStart, ttt] = ...
    simulateBCIFitts(initialTuning, D, alpha, beta, nDelaySteps, delT, nSimSteps);

disp(['With a matched, optimized decoder, mean trial time is ' num2str(mean(ttt)) ' s']);

%%
%Simulate a change in neural tuning (specifically, change the PDs only, making sure the total magnitude of tuning is the same)
newTuning = initialTuning;
newPD_component = randn(nUnits,2);
[Q, R] = qr([initialTuning(:,2:3), newPD_component]);
newPD_component = Q(:,3:4);

newTuning(:,2:3) = initialTuning(:,2:3)*(0.3) + newPD_component*(sqrt(1-0.3^2));

%%
%Simulate BCI performance under this change
[posTraj_new, velTraj_new, rawDecTraj_new, conTraj_new, targTraj_new, neuralTraj_new, trialStart_new, ttt_new] = ...
    simulateBCIFitts(newTuning, D, alpha, beta, nDelaySteps, delT, nSimSteps);

disp(['With changed tuning and a mismatched decoder, the mean trial time is ' num2str(mean(ttt_new)) ' s']);

%%
%This change in PDs effectively decrease the gain of the decoder, since the
%tuning strength in the decoder subspace has decreased (factor of 0.3).
%Here, we do a control to confirm that performance can't be restored simply
%by increasing the gain by (1/0.3) and using the original decoder subspace.
[posTraj_control, velTraj_control, rawDecTraj_control, conTraj_control, targTraj_control, neuralTraj_control, trialStart_control, ttt_control] = ...
    simulateBCIFitts(newTuning, D, alpha, beta/0.3, nDelaySteps, delT, nSimSteps);

disp(['Control: With changed tuning and a mismatched decoder with restored gain, the mean trial time is ' num2str(mean(ttt_new)) ' s']);

%%
%Here we se a simple HMM to infer the user's intended targets from the raw
%(unsmoothed) decoded velocity vectors & cursor positions alone.

%Each HMM state corresponds to one target location on a grid. First, get
%the grid of target locations.
gridSize = 20;
[X_loc,Y_loc] = meshgrid(linspace(-0.5, 0.5, gridSize), linspace(-0.5, 0.5, gridSize));
targLocs = [X_loc(:), Y_loc(:)];

stayProb = 0.9999;
nStates = gridSize^2;

%Define the state transition matrix, which has a large weight on the
%diagonal and small, uniform transition probabilities to other targets.
stateTrans = eye(nStates)*stayProb;
for x=1:nStates
    idx = setdiff(1:nStates, x);
    stateTrans(x,idx) = (1-stayProb)/(nStates-1);
end

pStateStart = zeros(nStates,1) + 1/nStates;

%Precision parameter for the von mises distribution.
vmKappa = 2;

%Infer traget locations from the raw decoder output and cursor positions
%using the viterbi algorithm (finds most likely sequence) and the
%forwards/backwards algorithm (to find the probabilities). These are custom
%functions that I made by modifying the MATLAB hmm routines (hmmviterbi & hmmdecode). 
[targStates, logP] = hmmviterbi_vonmises(rawDecTraj_new, stateTrans, targLocs, posTraj_new, pStateStart, vmKappa);
[pTargState, pSeq] = hmmdecode_vonmises(rawDecTraj_new, stateTrans, targLocs, posTraj_new, pStateStart, vmKappa);

%We can find time periods of high certainty, which may be of interest.
maxProb = max(pTargState);
highProbIdx = find(maxProb>0.8);

%See how well the inferred target locations match the true target
%locations.
inferredTargLoc = targLocs(targStates,:);

disp('Correlation between inferred target locations and true locations:');
disp(corr(targTraj_new, inferredTargLoc));

disp('Correlation between inferred target locations and true locations for periods of high certainty:');
disp(corr(targTraj_new(highProbIdx,:), inferredTargLoc(highProbIdx,:)));

figure; 
hold on;
plot(targTraj_new); 
plot(inferredTargLoc,'--');
legend({'True Target X','True Target Y','Inferred Target X','Inferred Target Y'});
xlabel('Time Step');
ylabel('X & Y Target Locations');

%%
%Now recalibrate the decoder based on the inferred targets.
inferredPosErr = inferredTargLoc - posTraj_new;
D_new = [ones(length(neuralTraj_new),1), neuralTraj_new] \ inferredPosErr;
decVec_new = [ones(size(neuralTraj_new,1),1), neuralTraj_new] * D_new;

%Important: normalize the decoder so that D_new decoders vectors of magnitude 1 when far from the
%target. This will restore the original optimal gain.
inferredTargDist = sqrt(sum(inferredPosErr.^2,2));
inferredTargDir = inferredPosErr ./ inferredTargDist;
farIdx = find(inferredTargDist>0.4);
projVec = sum(decVec_new(farIdx,:) .* inferredTargDir(farIdx,:),2);

D_new = D_new / mean(projVec);

%%
%Simulate BCI performance with the new decoder
[posTraj_recal, velTraj_recal, rawDecTraj_recal, conTraj_recal, targTraj_recal, neuralTraj_recal, trialStart_recal, ttt_recal] = ...
    simulateBCIFitts(newTuning, D_new, alpha, beta, nDelaySteps, delT, nSimSteps);

disp(['Recalibrating the decoder with inferred HMM targets, the mean trial time is ' num2str(mean(ttt_recal)) ' s']);

%%
%Now do the same thing, but using the true targets, so we can compare to
%the performance of supervised recalibration.
truePosErr = targTraj_new - posTraj_new;
D_supervised = [ones(length(neuralTraj_new),1), neuralTraj_new] \ truePosErr;
decVec_trueControl = [ones(size(neuralTraj_new,1),1), neuralTraj_new] * D_supervised;

trueTargDist = sqrt(sum(truePosErr.^2,2));
trueTargDir = truePosErr ./ trueTargDist;
farIdx = find(trueTargDist>0.4);
projVec = sum(decVec_trueControl(farIdx,:) .* trueTargDir(farIdx,:),2);

D_supervised = D_supervised / mean(projVec);

[posTraj_recal, velTraj_recal, rawDecTraj_recal, conTraj_recal, targTraj_recal, neuralTraj_recal, trialStart_recal, ttt_recal_super] = ...
    simulateBCIFitts(newTuning, D_supervised, alpha, beta, nDelaySteps, delT, nSimSteps);

disp(['Recalibrating the decoder with the true targets, the mean trial time is ' num2str(mean(ttt_recal_super)) ' s']);

%%
%Summarize the performance for all 4 relevant conditions: original
%performance, performance when the tuning changes (but the decoder
%doesn't), performance with the HMM-powered unsupervised recalibration, and
%performance with supervised recalibration. We plot means and 95% CIs.

[mu_original,SIGMAHAT,ci_original,SIGMACI] = normfit(ttt);
[mu_mismatch,SIGMAHAT,ci_mismatch,SIGMACI] = normfit(ttt_new);
[mu_hmmRecal,SIGMAHAT,ci_hmmRecal,SIGMACI] = normfit(ttt_recal);
[mu_supervisedRecal,SIGMAHAT,ci_supervisedRecal,SIGMACI] = normfit(ttt_recal_super);

mus = {mu_original, mu_mismatch, mu_hmmRecal, mu_supervisedRecal};
cis = {ci_original, ci_mismatch, ci_hmmRecal, ci_supervisedRecal};

labels = {'Original','Tuning Change','HMM Inference\newlineRecalibration','Supervised\newlineRecalibration'};

figure
hold on;
for x=1:length(mus)
    plot(x, mus{x}, 'o', 'Color', lines(1),'LineWidth',2);
    plot([x,x], cis{x}, '-', 'Color', lines(1),'LineWidth',2);
end
xlim([0.5, length(mus)+0.5]);
ylim([0,max([mus{:}])+0.2]);
ylabel('Mean Trial Time (s)');
set(gca,'XTick',1:length(mus),'XTickLabel',labels,'XTickLabelRotation',45,'FontSize',12,'LineWidth',2);
