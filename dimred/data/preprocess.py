import numpy as np
import functools as fc
from functools import partial

def MinMaxScalar(x):
    ## x is a one dimensional array 
    x  -= x.min()
    xmax = x.max()
    return x/xmax

def MeanMaxScalar(x):
    ## x is a feature set:
    x -= x.mean()
    return x/x.max()

def ZeroMeanScalar(x):
    ## x is a one dimensional array 
    x -= x.mean()
    vx = x.var()
    if vx == 0:
        print("Divide by zero Error")
    return x/vx

def NoScalar(x):
    return x/x.max()

def forward(x,substract,divide):
    x -= substract
    if any(divide) == 0:
        print("Divide by zero Error")
    else:
        x = x/divide
    return x

def reverse(x,substract,divide):
    x = x*divide
    x += substract
    return x


class Scalar:
    def __init__(self,scale_fun=MinMaxScalar) -> None:
        ## shape of input data is sklearn format (nsampes, nfeats)
        
        self.scale_fun = scale_fun
        pass
    
    def fit(self,dat):
        self.subs = np.min(dat,axis=0)
        self.divs = np.max(dat,axis=0)
        self.divs -= self.subs

    def transform(self,dat):
        return forward(dat)


    def fit_transform(self,dat):
        ## shape of X is sklearn format (nsampes, nfeats)
        return np.array([self.scale_fun(x) for x in dat.T]).T
        
    def transform2(self,dat):
        return reverse(dat)

def scaleData(tdat,scalar=ZeroMeanScalar,threshold=1e-11):
    ## input data usually has to be transped
    ## so outermost index corresp to feat
    adat = []
    ignor = []
    for i,a in enumerate(tdat.T):
        if featIgnore(a,eta=threshold):
            ignor.append(i)  ## ignore is decided using athresh
        else:
            adat.append(scalar(a))
    print("Ignored Features are",ignor)
    return np.array(adat).T



scaleMax = lambda dat : np.array([MinMaxScalar(x) for x in dat.T]).T
scaleStd = lambda dat : np.array([ZeroMeanScalar(x) for x in dat.T]).T
scaleAvg = lambda dat : np.array([MeanMaxScalar(x) for x in dat.T]).T

### TODO
### ext. scalar class with fun as args
# class 
