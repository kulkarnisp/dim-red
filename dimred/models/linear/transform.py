# utilities used in kurtosis analysis
import numpy as np
import functools as fc

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

def co_variance(X,bias=0):
    nx,i = X.shape ## X is input data matrix 
    ans = np.zeros((i,i),dtype=float)
    for a in X:
        ans += np.outer(a,a)
    return ans/(nx-bias)

def ex_variance(cm):
    i,_ = cm.shape
    fv = np.outer(cm,cm)
    return fv.reshape(i**3,i)

def ra_kurtosis(X,bias=0):
    nx,i = X.shape
    ans = np.zeros((i**3,i))
    for a in X:
        ans += np.outer(np.outer(np.outer(a,a),a),a)
    return ans/(nx-bias)

def co_kurtosis(rand_mat,bias=0):
    ck = ra_kurtosis(rand_mat,bias)
    cm = co_variance(rand_mat,bias)
    ev = ex_variance(cm)
    return ck- 3*ev



class Kurtosis:
    def __init__(self,n_retain=4,mom_path=None) -> None:
        self.n_retain = n_retain
        
        # pass

    # def recallMoments(self):

    def fit(self, X, moment=co_variance):
        self.cm = moment(X)
        u,s,v = np.linalg.svd(cm,full_matrices=False)
        self.vectors = u[:self.n_retain]
        self.values = s[:self.n_retain]
        
    def transform(self,X):
        return np.dot(self.vectors,X)

    # TodOdef transform@
    
    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)

    # def plot_plane(self,X):
    #     xr = self.fit_transform(X)[:,:2]
    #     plt.scatter(*xr.T)


    # def plot_embedding():

    # def decode(self):
#   