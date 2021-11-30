import numpy as np

def accuracy_score(x,y):
    return sum(np.abs(x-y))


def mean_sq_error(x,y):
    return np.mean((x-y)**2)

def mean_abs_error(x,y):
    return np.mean(abs(x-y))

