import imp
import numpy as np

def accuracy_score(x,y):
    return sum(np.abs(x-y))


def mean_sq_error(x,y):
    return np.mean((x-y)**2)

def mean_abs_error(x,y):
    return np.mean(abs(x-y))

def percent_err(x1,x2):
    err = abs(x1-x2)/x1*100
    return np.mean(err,axis=0)

def abs_err(x1,x2):
    # mmax = x1.mean(axis=0)
    # mmax += 1.0*(mmax == 0) 
    errs = np.mean((x1-x2)**2,axis=0)
    return errs