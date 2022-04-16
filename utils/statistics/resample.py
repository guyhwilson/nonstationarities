import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
import re

#def bootstrapMean(datas, n_bootstraps, CI = 0.95)


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
    


    

