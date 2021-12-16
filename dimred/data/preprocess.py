from matplotlib.pyplot import axis
import numpy as np
import functools as fc
from functools import partial

# def MinMaxScalar(x):
#     ## x is a one dimensional array 
#     x  -= x.min()
#     xmax = x.max()
#     return x/xmax
# def ZeroMeanScalar(x):
#     ## x is a one dimensional array 
#     x -= x.mean()
#     vx = x.var()
#     if vx == 0:
#         print("Divide by zero Error")
#     return x/vx

def MinMaxScalar():
    ## x is a feature set:
    # x -= x.mean()
    # return x/x.max()
    numer = lambda x: np.min(x,axis=0)
    denom = lambda x: np.max(x,axis=0) - np.min(x,axis=0)
    return numer,denom


def StandardScalar():
    numer = lambda x: np.mean(x,axis=0)
    denom = lambda x: np.var(x,axis=0) 
    return numer,denom


def AvgMaxScalar():
    numer = lambda x: np.mean(x,axis=0)
    denom = lambda x: np.max(x,axis=0) - np.mean(x,axis=0)
    return numer,denom

def MaxAvgScalar():
    numer = lambda x: np.mean(x,axis=0)
    denom = lambda x: np.max(x,axis=0)
    return numer,denom

def NoScalar():
    numer = lambda x: np.zeros(x.shape[1])
    denom = lambda x: np.ones(x.shape[1])
    return numer,denom

def _forward(x,substract,divide):
    x -= substract
    # if any(divide) == 0:
    #     print("Divide by zero Error")
    # else:
    #     x = x/divide
    zids = divide!=0
    x[:,zids] /= divide[zids]  ## amaze babe
    return x

def _reverse(x,substract,divide):
    x = x*divide
    x += substract
    return x

scale_sanity = lambda x: print(f"Maxima is {x.max():.2f} \n Minima is {x.min():.2f} \n {x.shape}")

class Scalar:
    def __init__(self,scale_gen=MinMaxScalar()) -> None:
        ## scale gen is a scale funciton generator
        ## shape of input data is sklearn format (nsampes, nfeats)
        self.numer,self.denom = scale_gen
#         self.denom
        pass
    
    def fit(self,dat):
        self.subs = self.numer(dat)
        self.divs = self.denom(dat)
#         self.divs -= self.subs

    def transform(self,dat):
        return _forward(dat,self.subs,self.divs)


    def fit_transform(self,dat):
        self.fit(dat)
        return self.transform(dat)
        ## shape of X is sklearn format (nsampes, nfeats)
#         return np.array([self.scale_fun(x) for x in dat.T]).T
        
    def transform2(self,dat,keep_status=False):
        if keep_status:
            subs = self.numer(dat)
            divs = self.denom(dat)
        else:
            subs = self.subs
            divs = self.divs
        return _reverse(dat,subs,divs).copy()
        
# def scaleData(tdat,scalar=ZeroMeanScalar,threshold=1e-11):
#     ## input data usually has to be transped
#     ## so outermost index corresp to feat
#     adat = []
#     ignor = []
#     for i,a in enumerate(tdat.T):
#         if featIgnore(a,eta=threshold):
#             ignor.append(i)  ## ignore is decided using athresh
#         else:
#             adat.append(scalar(a))
#     print("Ignored Features are",ignor)
#     return np.array(adat).T


def scaleData(dat,scalar=MinMaxScalar):
    up,down = scalar()
    return _forward(dat,up,down)

# scaleMax = lambda dat : np.array([MinMaxScalar(x) for x in dat.T]).T
# scaleStd = lambda dat : np.array([ZeroMeanScalar(x) for x in dat.T]).T
# scaleAvg = lambda dat : np.array([MeanMaxScalar(x) for x in dat.T]).T

class Shaper:
    def __init__(self) -> None:
        pass

    def fit(self,xinput):
        self.original_shape = list(xinput.shape[:-1])
        self.nvs = xinput.shape[-1]

    def transform(self,xinput):
        # nvs = xinput.shape[-1]  ## reshapeing to oned array for cantera
        return xinput.reshape(-1,self.nvs)

    def fit_transform(self,xinput):
        self.fit(xinput)
        return self.transform(xinput)

    def transform2(self,xout):
        nvs = xout.shape[-1]
        testlist = self.original_shape.copy()
        testlist.append(nvs)
        return xout.reshape(testlist) #        return xinp


### TODO
### ext. scalar class with fun as args
# class 
