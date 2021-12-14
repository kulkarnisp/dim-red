# utilities used in kurtosis analysis
import numpy as np
import functools as fc


def co_variance(X,bias=0):
    nx,i = X.shape ## X is input data matrix 
    ans = np.zeros((i,i),dtype=float)
    for a in X:
        ans += np.outer(a,a)
    return ans/(nx-bias)
co_variance.name = "co_variance" ## Abuse of function notations

def ex_variance(cm):
    i,_ = cm.shape
    fv = np.outer(cm,cm)
    return fv.reshape(i,i**3)

def ra_kurtosis(X,bias=0):
    nx,i = X.shape
    ans = np.zeros((i**3,i))
    for a in X:
        ans += np.outer(np.outer(np.outer(a,a),a),a)
    return ans.T/(nx-bias)
ra_kurtosis.name="Raw_kurtosis"

def co_kurtosis(rand_mat,bias=0):
    ck = ra_kurtosis(rand_mat,bias)
    cm = co_variance(rand_mat,bias)
    ev = ex_variance(cm)
    return ck- 3*ev
co_kurtosis.name = "co_kurtosis" 


# def outer_Variance()

def val_kurtosis(xscaled):
    shape = xscaled.shape
    n = shape[0]
    nvar = shape[1]
    CK = np.zeros((nvar, nvar, nvar, nvar))

    for i in range(0,(nvar)):
        for j in range(0,(nvar)):
            for k in range(0,(nvar)):
                for l in range(0,(nvar)):
                    CK[i,j,k,l] = (np.sum(xscaled[:,i]*xscaled[:,j]*xscaled[:,k]*xscaled[:,l]))
# #         print(i,j)
    CK=CK/n

    # CK = ra_kurtosis(xscaled)
    CV= co_variance(xscaled)
    
    # CE = np.zeros((nvar, nvar, nvar, nvar))
    for i in range(0,(nvar)):
        for j in range(0,(nvar)):
            for k in range(0,(nvar)):
                for l in range(0,(nvar)):
                    CK[i,j,k,l] = (CK[i,j,k,l] - CV[i,j]*CV[k,l] - CV[i,k]*CV[j,l] - CV[i,l]*CV[j,k])
    
    CK_m = CK.reshape(nvar, nvar*nvar*nvar)
    return CK_m

val_kurtosis.name="val kurtosis"


class Kurtosis:
    def __init__(self,n_retain=4,mom_path=None) -> None:
        self.n_retain = n_retain
        self.mom_path = mom_path
        # pass

    def recallMoments(self):
        ## todo add sanity checks, file exists ! is valid?
        self.cm = np.load(self.mom_path)

    def fit(self, X, moment=co_variance):
        if self.mom_path!=None:
            self.recallMoments()
        else:
            self.cm = moment(X)
        u,s,v = np.linalg.svd(self.cm,full_matrices=False)
        self.u = u.T
        self.s = s

        
    def transform(self,X):
        self.vectors = self.u[:self.n_retain]
        self.values = self.s[:self.n_retain]
        return np.dot(X,self.vectors.T)

    def transform2(self,projection):
        # projection = self.transform(X)
        return np.dot(projection,self.vectors)
    
    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)


    # def decode(self):
#   