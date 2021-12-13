import imp
import numpy as np
from ..data.preprocess import scaleMax

def accuracy_score(x,y):
    return sum(np.abs(x-y))


def mean_sq_error(x,y):
    return np.mean((x-y)**2)

def mean_abs_error(x,y):
    return np.mean(abs(x-y))

def abs_err(x1,x2):
    x1 = scaleMax(x1)
    x2 = scaleMax(x2)
    return np.mean(np.abs(x1-x2),axis=0)