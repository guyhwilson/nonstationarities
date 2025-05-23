import numpy as np
import scipy

def pitman_morgan(X,Y, verbosity = 0):
    """
    Pitman-Morgan Test for the difference between correlated variances with paired samples.
     
    Args:
        :X,Y: 
            | ndarrays with data.
        :verbosity: 
            | 0, optional
            | If 1: print results. 
            
    Returns:
        :tval:
            | statistic
        :pval:
            | p-value
        :df:
            | degree of freedom.
        :ratio:
            | variance ratio var1/var2 (with var1 > var2).

    Note:
        1. Based on Gardner, R.C. (2001). Psychological Statistics Using SPSS for Windows. New Jersey, Prentice Hall.
        2. Python port from matlab code by Janne Kauttonen (https://nl.mathworks.com/matlabcentral/fileexchange/67910-pitmanmorgantest-x-y; accessed Sep 26, 2019)
    """
    
    N = X.shape[0]
    var1, var2 = X.var(axis=0),Y.var(axis=0)
    cor = np.corrcoef(X,Y)[0,1]
    
    # must have var1 > var2:
    if var1 < var2:
        var1, var2 = var2, var1

    ratio = var1/var2
    
    # formulas from Garder (2001, p.57):
    numerator1_S1minusS2 = var1-var2
    numerator2_SQRTnminus2 = np.sqrt(N-2)
    numerator3 = numerator1_S1minusS2*numerator2_SQRTnminus2
    denominator1_4timesS1timesS2 = 4*var1*var2
    denominator2_rSquared = cor**2
    denominator3_1minusrSquared = 1.0 - denominator2_rSquared
    denominator4_4timesS1timesS2div1minusrSquared = denominator1_4timesS1timesS2*denominator3_1minusrSquared
    denominator5 = np.sqrt(denominator4_4timesS1timesS2div1minusrSquared)
    df = N-2
    if denominator5 == 0:
        denominator5 = _EPS
    tval = numerator3/denominator5
    
    # compute stats:
    p = 2*(1.0-scipy.stats.t.cdf(tval,df))
    if verbosity == 1:
        print('tval = {:1.4f}, df = {:1.1f}, p = {:1.4f}'.format(tval,df, p))

    return tval, p, df, ratio

def PermutationTest(group1, group2, permutations = 1000, tail = 'both'):
    '''
        Perform a permutation test of group means, where samples are independent. Inputs are:
        
        group1/2 (np.array) - holds test data 
        permutations (int)  - number of permutations to run
        tail (str)          - tails to be tested; options are 'left', 'right', and
                              'both'

        example: test two gaussians with equal variance and different means 
        
        group1 = np.random.normal(0, 1, 1000)
        group2 = np.random.normal(1, 1, 1000)
        pval   = permutationTest(group1, group)  
    
        TODO:
            - optimize code 
    '''
    
    null_distribution = np.zeros((permutations,))

    if len(group1.shape) > 1:
        if (group1.shape[1] > group1.shape[0]):
            group1  = np.transpose(group1)
    if len(group2.shape) > 1:
        if (group2.shape[1] > group2.shape[0]):
            group2  = np.transpose(group2)

    grouped_data  = np.concatenate((group1, group2))  
    grp1_len      = len(group1)
    
    for run in range(permutations):
        randomized_data = grouped_data[np.random.permutation(len(grouped_data))]
        shuffle1_mean   = np.mean(randomized_data[0:grp1_len])
        shuffle2_mean   = np.mean(randomized_data[(grp1_len + 1):])
    
        null_distribution[run] = shuffle1_mean - shuffle2_mean
    
    
    # calculate p-value 
    observed = np.mean(group1) - np.mean(group2)
    
    if (tail == 'left'):
        pval = (np.sum(observed >= null_distribution) + 1) / (permutations + 1)
    elif (tail == 'right'):
        pval = (np.sum(observed <= null_distribution) + 1) / (permutations + 1)
    else:
        pval = (np.sum(np.abs(observed) <= np.abs(null_distribution)) + 1) / (permutations + 1)

    return pval 
  

    
def signTest(group1, group2 = None, permutations = 1000, tail = 'both'):
    '''Paired-sample permutation test. Inputs are:
  
      group1/2 (np.array) - holds test data; if group2 is None, function assumes group1 holds pairwise differences
      permutations (int)  - number of permutations to run
      tail (str)          - tails to be tested; options are 'left', 'right', and
                            'both'
                            
      test design reference: 
        https://www.uvm.edu/~dhowell/StatPages/ResamplingWithR/RandomMatchedSample/RandomMatchedSampleR.html
    '''

    null_distribution = np.zeros((permutations,))
    
    if group2 is not None:
        if len(group1.shape) > 1:
            if (group1.shape[1] > group1.shape[0]):
                group1  = np.transpose(group1)
        if len(group2.shape) > 1:
            if (group2.shape[1] > group2.shape[0]):
                group2  = np.transpose(group2)
        pairwise_diffs     = group1 - group2   
        
    else: 
        pairwise_diffs = group1
        
        
    for run in range(permutations):
        randomized_data = pairwise_diffs * np.random.randint(2, size = len(group1))
        null_distribution[run] = np.mean(randomized_data)

    # calculate p-value 
    observed = np.mean(pairwise_diffs)

    if (tail == 'left'):
        pval = (np.sum(observed >= null_distribution) + 1) / (permutations + 1)
    elif (tail == 'right'):
        pval = (np.sum(observed <= null_distribution) + 1) / (permutations + 1)
    else:
        pval = (np.sum(np.abs(observed) <= np.abs(null_distribution)) + 1) / (permutations + 1)

    return pval, null_distribution