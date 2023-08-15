% Inputs:
%sessions_dir = 'C:\Users\ghwilson\Documents\SidData\';
%save_dir     = 'C:\Users\ghwilson\Documents\SidData\';

sessions_dir = '/oak/stanford/groups/shenoy/cortex/data/JenkinsC/expdata/decoder/R/';
save_dir     = '/oak/stanford/groups/shenoy/ghwilson/nonstationarities/monkey/';


%% Load FileLogger data and process using Frank/Eli's formatting
files = dir([sessions_dir, '*R_2012-08-03_1.mat']);

for i = 1:length(files)
    load([files(i).folder, '/', files(i).name]);
  
    
    % select subset of original R struct
    miniR      = struct;
    fields = ["cursorPos", "spikeRaster", "spikeRaster2", "trialNum", "isSuccessful", "startTrialParams"];
              
    for j = 1:length(fields)
        miniR.(fields(j)) = {R.(fields(j))};
    end
    
    subj       = {R.subject};
    save_fname = [save_dir, '/', subj{1}, '/'];
    if ~exist(save_fname, 'dir')
        mkdir(save_fname);
    end
    save([save_fname, files(i).name], 'miniR', '-v7');
    disp([save_fname, files(i).name])
end

%R = load(file)

%dataset = formatWest2DDataset(sess, saveDir, session_dir);
%save([saveDir filesep sess{1} '.mat'],'dataset');
