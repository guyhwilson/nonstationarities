import numpy as np



class RTI(object):
    """
    RTI model for unsupervised recalibration. 
    """
    
    def __init__(self, look_back, min_dist, min_time):
        '''Inputs are:
        
        '''
        
        assert min_dist >= 0, "min_dist parameter must be nonnegative"
        assert min_time >= 0, "min_time parameter must be nonnegative"
        assert look_back == -1 or look_back >= 0 , "look_back parameter must be nonnegative or -1"
        
        self.look_back = int(look_back) if look_back >= 0 else 'prior_click'
        self.min_dist  = min_dist
        self.min_time  = int(min_time)
            
    
    def label(self, neural, cursor, click_state, return_indices=False):
        '''Label data to provide estimated target states. Inputs are:
        
            neural (2D float)      - time x channels of neural activity
            cursor (2D float)      - time x 2 of cursor positions
            click_state (1D float) - binary indicating whether or not click occurred at 
                                     each timestep
            return_indices (bool)  - whether or not to return the timestamps included by RTI 
        '''
        
        clicks   = np.where(click_state == 1)[0]
        features = list()
        targets  = list()
        idxs     = list()
        
        for idx, i in enumerate(clicks):
            target = cursor[i, :]
            
            # look back at previous timepoints up to prior click
            if idx == 0:
                if self.look_back == 'prior_click':
                    start = 0
                else:
                    start = max(i - self.look_back, 0)
                
            if idx > 0:
                if self.look_back == 'prior_click':
                    start = clicks[idx-1]
                else:
                    start = max(i - self.look_back, clicks[idx-1])
                
            stop   = i - self.min_time + 1       # include only up to some window prior to click 
            
            # subselect time range of interest before click
            displacement_snippet = target - cursor[start:stop, :]
            distance_snippet     = np.linalg.norm(displacement_snippet, axis = 1)
            neural_snippet       = neural[start:stop, :]
            
            # now fine-tune that selection with some heuristics
            try:
                far_idx       = distance_snippet > self.min_dist  # distant from inferred target
                approach_idx  = np.gradient(distance_snippet) < 0 # cursor heading toward inferred target
                select_idx    = np.logical_and(far_idx, approach_idx)

                features.append(neural_snippet[select_idx, :])
                targets.append(displacement_snippet[select_idx, :])
                idxs.append(np.arange(start, stop)[select_idx])
            except:
                pass  # some edge cases where very little data that doesnt meet requirements 

        features  = np.vstack(features)
        targets   = np.vstack(targets)
        idxs      = np.concatenate(idxs)
        
        if return_indices:
            return features, targets, idxs
        else:
            return features, targets
