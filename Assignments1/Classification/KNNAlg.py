import numpy as np
import logging


def distance(data,center):
    return np.linalg.norm(data-center,ord=2,axis=1)


def KNN(candidate:np.ndarray,voters:np.ndarray,votersLabel:np.ndarray,neighborNumber=None):
    if neighborNumber==None:
        neighborNumber=min(candidate.shape[0],5)
    assert(candidate.shape[0]==1)
    dist=np.linalg.norm(voters-candidate,ord=2,axis=1)
    kNearstNeighborsIndex=np.argsort(dist)[:neighborNumber]
    votes=votersLabel[kNearstNeighborsIndex]
    countVotes=np.bincount(votes)
    return np.argmax(countVotes)