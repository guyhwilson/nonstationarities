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
        assert look_back >= 0, "look_back parameter must be nonnegative"
        
        self.look_back = int(look_back)
        self.min_dist  = min_dist
        self.min_time  = int(min_time)
    
    
    def label(self, neural, cursor, click_state):
        '''Label data to provide estimated target states.'''
        
        clicks   = np.where(click_state == 1)[0]
        features = list()
        targets  = list()
        
        for idx, i in enumerate(clicks):
            target = cursor[i, :]
            
            # look back at previous timepoints up to prior click
            if idx == 0:
                start = max(i - self.look_back, 0) 
            if idx > 0:
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
            except:
                pass  # some edge cases where very little data that doesnt meet requirements 

        features     = np.vstack(features)
        targets      = np.vstack(targets)
        
        return features, targets
