from matplotlib.pyplot import axis
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


def AvgMaxScalar(x):
    xmax = abs(x).max()
    x -= x.mean()
    return x/xmax


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
    # if any(divide) == 0:
    #     print("Divide by zero Error")
    # else:
    #     x = x/divide
    zids = divide!=0
    x[:,zids] /= divide[zids]  ## amaze babe
    return x

def reverse(x,substract,divide):
    x = x*divide
    x += substract
    return x

scale_sanity = lambda x: print(f"Maxima is {x.max():.2f} \n Minima is {x.min():.2f} \n {x.shape}")

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
        return forward(dat,self.subs,self.divs)


    def fit_transform(self,dat):
        self.fit(dat)
        return self.transform(dat)
        ## shape of X is sklearn format (nsampes, nfeats)
#         return np.array([self.scale_fun(x) for x in dat.T]).T
        
    def transform2(self,dat,keep_status=False):
        if keep_status:
            subs = dat.min(axis=0)
            divs = subs - dat.max(axis=1)

        return reverse(dat,self.subs,self.divs).copy()

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
