%% Select block to process; add relevant helper functions to path

% Inputs:
saveDir    = 'D:\T5_ClosedLoop\new';
session_dir = 'D:\T5_OfflineDailyClick'; 

session     = 't5.2021.07.26';
blocks      =  [2, 3 ,4];  % NOTE: assumes 0-indexing (i.e. there's a block 0)
% ----------------------------------------------

cd(session_dir)
addpath(genpath('C:\Users\ghwilson\Documents\projects\nptlbraingaterig\code\nptlDataExtraction\'))
addpath(genpath('C:\Users\ghwilson\Documents\projects\nptlbraingaterig\code\utilities\'))
addpath(genpath('C:\Users\ghwilson\Documents\projects\nptlbraingaterig\code\lib\'))
addpath(genpath('C:\Users\ghwilson\Documents\projects\nptlbraingaterig\code\analysis\Frank\'))
addpath(genpath('C:\Users\ghwilson\Documents\projects\nptlbraingaterig\code\analysis\Guy\nonstationarities\scripts\'))


%% Load FileLogger data and process using Frank/Eli's formatting

sess    = cell(2, 1);
sess{1} = session;
sess{2} = blocks;


dataset = formatWest2DDataset(sess, saveDir, session_dir);
save([saveDir filesep sess{1} '.mat'],'dataset');


%% Debugging
%{
R_blocks = cell(length(blocks), 1);

for i = 1:length(blocks)
    block       = blocks(i);
    stream      = parseDataDirectoryBlock(block); 
    R_blocks{i} = onlineR(stream);
end


R_struct = [R_blocks{:}]; 


%}
