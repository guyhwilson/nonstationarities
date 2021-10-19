
function out = processBlocks(blocks)
% Helper function for preprocessing block data before feeding into python scripts. Inputs are:
% 
%    blocks (list)      - list of blocks given by integer ID
%    save_dir (str)     - output .mat file save directory
%
% NOTE: needs to be called from just above the session folder

    % Inputs:
   % save_dir    = 'D:\T5_OfflineDailyClick\processed';
   %session_dir = 'D:\T5_OfflineDailyClick'; 

    %session     = 't5.2021.05.17';
    %blocks      =  [1, 4];  % NOTE: assumes 0-indexing (i.e. there's a block 0)
    % ----------------------------------------------
    session     = 'Session';
    save_dir    = 'E:\Session\Data\HMMrecal';
    session_dir = 'E:';
    root_dir    = 'E:\Session\Software\nptlbraingaterig\';
   % root_dir = 'C:\Users\ghwilson\Documents\projects\nptlbraingaterig\'

    addpath(genpath([root_dir, 'code\nptlDataExtraction\']));
    addpath(genpath([root_dir, 'code\utilities\']));
    addpath(genpath([root_dir, 'code\lib\']));
    addpath(genpath([root_dir, 'code\analysis\Frank\']));

    %% Load FileLogger data and process using Frank/Eli's formatting

    sess    = cell(2, 1);
    sess{1} = session;
    sess{2} = blocks;

    dataset    = formatWest2DDataset(sess, save_dir, session_dir);
    compressed = struct('blockList', [], 'blockNums', [], 'targetPos', [], 'cursorPos', [], 'TX', [], ...
                        'decVel', []);
    
    for fn = fieldnames(compressed)'
       disp(fn)
       compressed.(fn{1}) = dataset.(fn{1});
    end
    
  
    save([saveDir filesep sess{1} '_compressed.mat'],'compressed');





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
end