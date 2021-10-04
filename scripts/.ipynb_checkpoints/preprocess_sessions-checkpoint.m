%% Select block to process; add relevant helper functions to path

% Inputs:
save_dir    = 'D:\T5_OfflineDailyClick\processed';
session_dir = 'D:\T5_OfflineDailyClick'; 

session     = 't5.2021.05.17';
blocks      =  [1, 4];  % NOTE: assumes 0-indexing (i.e. there's a block 0)
% ----------------------------------------------

cd(session_dir)
addpath(genpath('C:\Users\ghwilson\Documents\projects\nptlbraingaterig\code\nptlDataExtraction\'))
addpath(genpath('C:\Users\ghwilson\Documents\projects\nptlbraingaterig\code\utilities\'))
addpath(genpath('C:\Users\ghwilson\Documents\projects\nptlbraingaterig\code\lib\'))
addpath(genpath('C:\Users\ghwilson\Documents\projects\nptlbraingaterig\code\analysis\Frank\'))

%% Load FileLogger data and process using Frank/Eli's formatting

sess    = cell(2, 1);
sess{1} = session;
sess{2} = blocks;


formatWest2DDataset(sess, save_dir, session_dir)





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
