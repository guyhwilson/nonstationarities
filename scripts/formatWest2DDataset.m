function dataset = formatWest2DDataset( sess, saveDir, sessDir )
%  Inputs are:
%      sess (2x1 cell array) - {1} = [session folder] 
%                              {2} = [desired blocks]
%      saveDir (str)         - path to save directory
%      sessDir (str)         - path to raw dataset directory
%
    
    blockList = sess{2};
    dataset.blockList = blockList;
    binMS = 20;
    
    %define dataset struct
    dataset.blockList = blockList;
    dataset.targetSize = [];
    dataset.cursorSize = [];
    dataset.blockNums = [];
    dataset.sysClock = [];
    dataset.nspClocks = [];
    dataset.cursorPos = [];
    dataset.targetPos = [];
    dataset.onTarget = [];
    dataset.spikePow = [];
    dataset.decVel = [];
    dataset.decClick = [];
    dataset.TX = [];
    dataset.TX_thresh = -3.5;
    dataset.decodingClick = [];
    dataset.trialEpochs = [];
    dataset.instructedDelays = [];
    dataset.intertrialPeriods = [];
    dataset.gameNames = [];
    dataset.isSuccessful = [];
    
    %first get RMS
    R = [];
    for b=1:length(blockList)
        disp(blockList(b));

        sessionDir = [sessDir filesep sess{1}];
        cd(sessionDir);
        stream  = parseDataDirectoryBlock([sessionDir filesep 'Data' filesep 'FileLogger' filesep num2str(blockList(b)) filesep]);
        tmpR    = onlineR(stream);
        fNames  = fieldnames(tmpR);
        keepIdx = strcmp(fNames, 'meanSquaredAcaus') | strcmp(fNames,'meanSquaredAcausChannel');
        tmpR    = rmfield(tmpR, fNames(~keepIdx));
        
        R = [R, tmpR];
    end
    
    rms = channelRMS(R);
    thresh = dataset.TX_thresh*rms;
    
    %now get other variables block by block (since the game may change from
    %block to block)
    globalLoopIdx = 1;
    for b=1:length(blockList)
        disp(blockList(b));

        sessionDir = [sessDir filesep sess{1}];
        cd(sessionDir);
        stream = parseDataDirectoryBlock([sessionDir filesep 'Data' filesep 'FileLogger' filesep num2str(blockList(b)) filesep]);
        R = onlineR(stream);
        
        nChans = size(R(1).minAcausSpikeBand,1);
        for t=1:length(R)
            R(t).spikeRaster = bsxfun(@lt, R(t).minAcausSpikeBand(1:96,:), thresh(1:96)');
            if nChans>96
                R(t).spikeRaster2 = bsxfun(@lt, R(t).minAcausSpikeBand(97:end,:), thresh(97:end)');
            end
        end
        
        %make spike raster by applying thresholds
        opts.filter = false;
        data = unrollR_generic(R, binMS, opts);

        nBlockLoops = length(data.targetPos);
        nLoops = length([R.clock]);
        targRad = zeros(nLoops,1);
        blockNum = zeros(nLoops,1);

        globalIdx = 1;
        for t=1:length(R)
            nTrlLoop = length(R(t).clock);
            currIdx = (globalIdx):(globalIdx+nTrlLoop-1);
            if isfield(R(t).startTrialParams,'targetDiameter')
                targRad(currIdx) = double(R(t).startTrialParams.targetDiameter/2);
            else
                targRad(currIdx) = 50;
            end
            blockNum(currIdx) = blockList(b);
            globalIdx = globalIdx + nTrlLoop;
        end

        dataset.targetSize = [dataset.targetSize; targRad(1:binMS:end)];
        dataset.cursorSize = [dataset.cursorSize; zeros(nBlockLoops,1) + 25];
        dataset.blockNums = [dataset.blockNums; repmat(blockList(b), nBlockLoops, 1)];

        if isfield(R,'firstCerebusTime')
            nspClocks = [R.firstCerebusTime]';
            dataset.nspClocks = [dataset.nspClocks; nspClocks(1:binMS:end,:)];
        end
        
        sysClock = [R.clock]';
        dataset.sysClock = [dataset.sysClock; sysClock(1:binMS:end)];
        
        dataset.cursorPos = [dataset.cursorPos; data.cursorPos(:,1:2)];
        dataset.targetPos = [dataset.targetPos; data.targetPos(:,1:2)];

        if isfield(R,'clickState')
            clickState = [R.clickState]';
        else
            clickState = [];
            for t=1:length(R)
                clickState = [clickState; R(t).decoderC.discreteStateLikelihoods(end,:)'];
            end
        end    
        clickState = double(clickState);
        xk = [R.xk]';
        if strcmp(sess{1}(1:2),'t6')
            dataset.decVel = [dataset.decVel; xk(1:binMS:end,[1 2])];
        else
            dataset.decVel = [dataset.decVel; xk(1:binMS:end,[2 4])];
        end
        dataset.decClick = [dataset.decClick; clickState(1:binMS:end)];
        dataset.TX = [dataset.TX; data.spikes];
        if isfield(data,'hlfp') && ~all(data.hlfp(:)==0)
            dataset.spikePow = [dataset.spikePow; data.hlfp];
        end
        
        dataset.decodingClick = [dataset.decodingClick; any(clickState(1:binMS:end)~=0)];
        
        if isfield(R(1),'overCuedTarget')
            ot = [R.overCuedTarget]';
            ot = ot(1:binMS:end)>0;
            dataset.onTarget = [dataset.onTarget; ot];
        else
            targDist = matVecMag(data.targetPos - data.cursorPos,2);
            dataset.onTarget = [dataset.onTarget; targDist < (targRad(1:binMS:end) + 25)];
        end
        
        if strcmp(R(1).taskDetails.taskName,'keyboard')
            %grid task has misleading timeTargetOn
            newTrialEpochs = [[1; data.reachEvents(1:(end-1),3)+1], data.reachEvents(:,3)];
            newInstructedDelays = nan(length(newTrialEpochs),2);
            newIntertrialPeriods = nan(length(newTrialEpochs),2);
        else
            newTrialEpochs = data.reachEvents(:,[2 3]);
            newInstructedDelays = data.reachEvents(:,[1 2]);

            tmpStart = data.reachEvents(:,2);
            hasDelay = ~isnan(data.reachEvents(:,1));
            tmpStart(hasDelay) = data.reachEvents(hasDelay,1);
            newIntertrialPeriods = [[1; data.reachEvents(1:(end-1),3)], tmpStart];
            newInstructedDelays(~hasDelay,:) = NaN;
        end

        if isfield(R,'isSuccessful')
            dataset.isSuccessful = [dataset.isSuccessful; [R.isSuccessful]'];
        else
            dataset.isSuccessful = [dataset.isSuccessful; nan(length(R),1)];
        end
        dataset.trialEpochs = [dataset.trialEpochs; newTrialEpochs+globalLoopIdx];
        dataset.instructedDelays = [dataset.instructedDelays; newInstructedDelays+globalLoopIdx];
        dataset.intertrialPeriods = [dataset.intertrialPeriods; newIntertrialPeriods+globalLoopIdx];
        globalLoopIdx = globalLoopIdx + nBlockLoops;
        
        dataset.gameNames = [dataset.gameNames; {R(1).taskDetails.taskName}];
    end
    
   % save([saveDir filesep sess{1} '.mat'],'dataset');
end

