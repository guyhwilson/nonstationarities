import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
import re

import os
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt


""" Author: Benyamin Meschede-Krasa 
cross validated distance, based on https://github.com/fwillett/cvVectorStats/blob/master/cvDistance.m """

def cvDistance(class0,class1,subtractMean=False, CIMode='none',CIAlpha=0.05): #TODO implement CI
    """Estimate the distance between two distributions
    Parameters
    ----------
    class0 : ndarray (nTrials,nFeatures)
        samples from distributions to be compared 
    class1 : _type_
        _description_
    subtractMean : bool, optional
        If subtractMean is true, this will center each vector
        before computing the size of the difference, by default False
    CIMode : str
        method for computing confidence intervals. Currently only 'jackknife'
        is implmented
    CIAlpha : float
        alpha for confidence interval. Default is 0.05 which give the 95%
        confidence interval
    Returns
    -------
    squaredDistance : float
        cross-validated estimate of squared distance between class 1 and 2
    euclideanDistance : float
        cross-validated estimate of euclidean distance between class 1 and 2
    CI : ndarray(2,2)
        confidence intervals for squaredDistance (col 0) and euclideanDistance
        (col 1)
    """
    class0 = np.array(class0)
    class1 = np.array(class1)

    assert class0.shape == class1.shape, "Classes must have same shape, different numebrs of trials not implemented yet" #TODO implement different trial numebr for classes

    nTrials, nFeatures = class0.shape
    squaredDistanceEstimates=np.zeros([nTrials,1])

    for x in range(nTrials):
        bigSetIdx = list(range(nTrials))
        smallSetIndex = bigSetIdx.pop(x)

        meanDiff_bigSet = np.mean(class0[bigSetIdx,:] - class1[bigSetIdx,:],axis=0)
        meanDiff_smallSet = class0[smallSetIndex,:] - class1[smallSetIndex,:]
        if subtractMean:
            squaredDistanceEstimates[x] = np.dot(meanDiff_bigSet-np.mean(meanDiff_bigSet)),(meanDiff_smallSet-np.mean(meanDiff_smallSet).transpose())
        else:
            squaredDistanceEstimates[x] = np.dot(meanDiff_bigSet,meanDiff_smallSet.transpose())
    
    squaredDistance = np.mean(squaredDistanceEstimates)
    euclideanDistance = np.sign(squaredDistance)*np.sqrt(np.abs(squaredDistance))
    
    if CIMode == 'jackknife':
        wrapperFun = lambda x,y : cvDistance(x,y,subtractMean=subtractMean)
        [CI, CIDistribution] = cvJackknifeCI([squaredDistance, euclideanDistance], wrapperFun, [class0, class1], CIAlpha)
    elif CIMode == 'none':
        CI = []
        CIDistribution = []
    else:
        raise ValueError(f"CIMode {CIMode} not implemented or is invalid. select from ['jackknife','none']")

    return squaredDistance, euclideanDistance, CI, CIDistribution 


def cvJackknifeCI(fullDataStatistic, dataFun, dataTrials, alpha):
    """compute confidence intervals for cv statistic
    Parameters
    ----------
    fullDataStatistic : list
        list of statistics computed from `dataTrials` 
    dataFun : func
        callable function that transforms `dataTrials` to the
        statistic
    dataTrials : array (n_classes, n_trials, n_features)
        list of data from classes used to compute `fullDataStatistic`
    alpha : float
        alpha for confidence interval coverage (e.g. 0.05 for 95%CI)
    Returns
    -------
    CI : array (n_statistics, 2)
        upper and lower bounds for each statistic
    jacks : array (n_folds, n_statistics)
        folds from jackknifing (loo)
    """

    # NOTE: implementation only supports data cells with same numbers of trials unlike original implementation
    nFolds = dataTrials[0].shape[0] # Leave one trial out cross validation
    folds = np.arange(nFolds)
    jacks = np.zeros([nFolds, len(fullDataStatistic)]) 
    for foldIdx in folds:
        deleteTrials = [list(dataTrial) for dataTrial in dataTrials]
        for x in range(len(deleteTrials)):
            deleteTrials[x].pop(foldIdx)
        jacks[foldIdx,:] = dataFun(*deleteTrials)[:2]

    ps = nFolds*np.array(fullDataStatistic) - (nFolds-1)*jacks
    v  = np.var(ps,axis=0) 
    
    multiplier = norm.ppf((1-alpha/2), 0, 1)
    CI = np.array([(fullDataStatistic - multiplier*np.sqrt(v/nFolds)), (fullDataStatistic + multiplier*np.sqrt(v/nFolds))])
    return CI, jacks


def cvCorr( class1, class2, CIMode = 'None', CIAlpha = 0.05, CIResamples = 10000):
    '''
    class1 and class2 are N x D matrices, where D is the number of
    dimensions and N is the number of samples
    
    this function estimates the correlation between the mean vectors of
    class1 and class2.
    
    CIMode can be none, bootCentered, bootPercentile, or jackknife
    
    CIAlpha sets the coverage of the confidence interval to
    100*(1-CIAlpha) percent
    
    CIResamples sets the number of bootstrap resamples, if using bootstrap
    mode (as opposed to jackknife)
    
    CIDistribution is the distribution of bootstrap statistics or
    jackknife leave-one-out statistics '''
    
    
    unbiasedMag1 = cvDistance( class1, np.zeros(class1.shape), True )[0]
    unbiasedMag2 = cvDistance( class2, np.zeros(class2.shape), True )[0]

    mn1 = np.mean(class1)
    mn2 = np.mean(class2)
    cvCorrEst = (mn1-np.mean(mn1)).dot((mn2-np.mean(mn2)).T)/(unbiasedMag1 * unbiasedMag2)
    
    #compute confidence interval if requensted    
    if CIMode is not None:
        raise NotImplementedError
        #wrapperFun = @(x,y)(cvCorr(x,y));
        #[CI, CIDistribution] = cvCI(cvCorrEst, wrapperFun, {class1, class2}, CIMode, CIAlpha, CIResamples);
    else:
        CI = []
        CIDistribution = []
    
    return cvCorrEst, CI, CIDistribution




def bootstrap_LinearRegression(x, y, regressor = LinearRegression(), n_bootstraps = 1000, random_state = None):
    '''Bootstrap linear regression estimate of relationship between sets of variables. Inputs are:

    x (2D array)               - samples x features array of predictions
    y (1D array)               - samples x 1 array of targets
    regressor (sklearn object) - regression method; defaults to LinearRegression()
    n_bootstraps (int)         - number of bootstrap iterations to run

    '''

    n_samples, n_features = x.shape
    bootstrap_coefs       = np.zeros((n_bootstraps, n_features))
    bootstrap_means       = np.zeros((n_bootstraps, n_features))
    mean_lm               = regressor.fit(x, y)

    for i in range(n_bootstraps):
        sample_index = np.random.choice(n_samples, n_samples, replace = True)
        x_sample     = x[sample_index, :]
        y_sample     = y[sample_index]
        lm           = regressor.fit(x_sample, y_sample)

        bootstrap_means[i, :] = lm.intercept_
        bootstrap_coefs[i, :] = lm.coef_

    return mean_lm, bootstrap_means, bootstrap_coefs



def makeSessionPairGraph(scores_df, lims = [0, np.inf], int_encoded = False):
    
    if not int_encoded:
        pat   = '\d\d\d\d\.\d\d\.\d\d'
        dates = np.asarray([re.findall(pat, x) for x in scores_df['file']])
    else:
        dates = np.asarray([x for x in scores_df['file']])

    subselect = np.logical_and(scores_df['days_apart'] >= lims[0], scores_df['days_apart'] < lims[1] )
    
    dates = dates[subselect]
    
    # number of vertices
    V = len(dates)
    E = list()
    
    for i, date_1 in enumerate(dates):
        for j, date_2 in enumerate(dates):
            if j > i:
                if len(np.intersect1d(date_1, date_2)) > 0:
                    E.append((i, j))
                    
    # Constructs Graph as a dictionary of the following format:
    # graph[VertexNumber V] = list[Neighbors of Vertex V]
    graph = dict([])
    for i in range(len(E)):
        v1, v2 = E[i]

        if(v1 not in graph):
            graph[v1] = []
        if(v2 not in graph):
            graph[v2] = []

        graph[v1].append(v2)
        graph[v2].append(v1)
        
    return graph, np.where(subselect)[0]



def graphSets(graph):
      
    # Base Case - Given Graph 
    # has no nodes
    if(len(graph) == 0):
        return []
     
    # Base Case - Given Graph
    # has 1 node
    if(len(graph) == 1):
        return [list(graph.keys())[0]]
      
    # Select a vertex from the graph
    vCurrent = list(graph.keys())[0]
      
    # Case 1 - Proceed removing
    # the selected vertex
    # from the Maximal Set
    graph2 = dict(graph)
      
    # Delete current vertex 
    # from the Graph
    del graph2[vCurrent]
      
    # Recursive call - Gets 
    # Maximal Set,
    # assuming current Vertex 
    # not selected
    res1 = graphSets(graph2)
      
    # Case 2 - Proceed considering
    # the selected vertex as part
    # of the Maximal Set
  
    # Loop through its neighbours
    for v in graph[vCurrent]:
          
        # Delete neighbor from 
        # the current subgraph
        if(v in graph2):
            del graph2[v]
      
    # This result set contains VFirst, and the result of recursive call assuming 
    # neighbors of vFirst are not selected
    res2 = [vCurrent] + graphSets(graph2)
      
    # Our final result is the one which is bigger, return it
    if(len(res1) > len(res2)):
        return res1
    return res2
    


    

