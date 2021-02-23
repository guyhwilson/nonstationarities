%%
% Code by Frank Willett

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
