# For converting .mat files into python structs.

import numpy as np
from scipy.io import loadmat
import os
from copy import deepcopy
from datetime import date
import firingrate


def daysBetween(date_a, date_b):
    '''Number of days between two sessions. Input format is:
    
        year.month.day - e.g. '2020.0.113'
    '''
    date_a = date(*[int(x) for x in date_a.split('.')])
    date_b = date(*[int(x) for x in date_b.split('.')])
    days   = (date_a - date_b).days
    
    return np.abs(days) 


class DataStruct(object):
    """
    Generates a simplified R struct from cursor data.
    """
    def __init__(self, file, alignScreens = False, causal_filter = 0):
        
        assert isinstance(file, str), "<file> must be a string"
        assert isinstance(alignScreens, bool), "<alignScreens> must be bool"
        
        dat                = loadmat(file)['dataset'][0][0]
        self.date          = file.split('t5.')[1].split('.mat')[0]

        self.blockList            = dat[0][0]    # contains block labels 
        self.gameName             = dat[18]   # of same length as blocklist: contains string label of task for each block
        self.cursorPos_continuous = dat[6].astype('float')
        self.targetPos_continuous = dat[7].astype('float')
        self.decClick_continuous  = np.concatenate(dat[11])
        self.onTarget_continuous  = np.concatenate(dat[8])
        self.TX_thresh            = dat[13]
        self.trialEpochs          = dat[15] - 1  # account for MATLAB's 1-indexing
        self.trialEpochs[:, 1]   -= 1            # account for MATLAB's inclusive indexing 
        self.sysClock             = dat[4]
        self.nspClocks            = dat[5]
        self.decVel               = dat[10]
        self.n_trials             = self.trialEpochs.shape[0]
         
        # now load in neural data and smooth if requested:
        if causal_filter > 0:
            self.TX_continuous = firingrate.gaussian_filter1d(dat[12].astype('float'), sigma = causal_filter, axis = 0, causal = True)
        else:
            self.TX_continuous = dat[12].astype('float')
        self.n_channels = self.TX_continuous.shape[1] 

        TX         = list()
        blockNums  = list()
        trialType  = list()
        isSuccessful = list()

        targetSize, cursorSize = list(), list()
        cursorPos, targetPos   = list(), list()
        decClick               = list()
        onTarget               = list()
        
        keep = list()
        for i in range(self.n_trials):
            start, stop = self.trialEpochs[i, :] 
            if stop - start > 5:         # hard coded hack - toss out weird short trials
                keep.append(i)

                targetSize.append(dat[1][start][0])
                cursorSize.append(dat[2][start][0])
                blockNums.append(dat[3][start][0])

                trialType.append(self.gameName[np.where(self.blockList == blockNums[-1])[0]][0][0][0])
                isSuccessful.append(dat[19][i][0])

                TX.append(deepcopy(self.TX_continuous[start:stop, :]))
                cursorPos.append(deepcopy(self.cursorPos_continuous[start:stop, :]))
                targetPos.append(deepcopy(self.targetPos_continuous[start:stop, :]))
                decClick.append(deepcopy(self.decClick_continuous[start:stop]))
                onTarget.append(deepcopy(self.onTarget_continuous[start:stop]))

        self.TX           = TX
        self.targetSize   = np.asarray(targetSize, dtype = 'object')
        self.cursorSize   = np.asarray(cursorSize, dtype = 'object')
        self.targetPos    = targetPos
        self.cursorPos    = cursorPos
        self.decClick     = np.asarray(decClick, dtype = 'object')
        self.onTarget     = np.asarray(onTarget, dtype = 'object')
        self.blockNums    = np.asarray(blockNums, dtype = 'object')
        self.IsSuccessful = np.asarray(isSuccessful, dtype = 'object')
        self.trialType    = np.asarray(trialType, dtype = 'object')
        
        self.trialEpochs  = self.trialEpochs[np.asarray(keep), :]

        self.screenAligned = False
        if alignScreens:
            self.alignTaskScreens()
            

    def alignTaskScreens(self):
        '''Realign screen positioning across different tasks into a common reference frame by centering all
           screens at (0, 0). Inputs are:

            struct (DataStruct) - session data to use 
        '''
        
        if not self.screenAligned:
            screen_realignments = np.load('../utils/misc_data/screen_realignments.npy', allow_pickle = True).item()
            
            for i, task in enumerate(self.trialType):
                self.cursorPos[i] -= screen_realignments[task]
                self.targetPos[i] -= screen_realignments[task]
        else:
            print('Screens already aligned across tasks. Skipping.')