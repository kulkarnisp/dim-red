
from numba import jit,vectorize,njit

import numpy as np
import functools as fc
import os
import time
# from tqdm.notebook import tqdm_notebook as tqdm
from tqdm import tqdm

# tqdm  = lambda x : x

# def co_variance(X,bias=0):
#     nx,i = X.shape ## X is input data matrix 
#     # ans = np.zeros((i,i))#,dtype=float)
#     # for a in X:
#     #     ans += np.outer(a,a)
#     ans = X.T.dot(X)
#     return ans/(nx-bias)

def co_variance(x,bias=0):
    return x.T.dot(x)/(x.shape[0]-bias)
co_variance.name = "co_variance" ## Abuse of function notations


def ex_variance(cm):
    i,_ = cm.shape
    fv = np.outer(cm,cm)
    return fv.reshape(i,i**3)


def ra_kurtosis(X,bias=0):
    nx,i = X.shape
    ans = np.zeros((i**3,i))
    looper = tqdm(X)
    for a in looper:
        ans += np.outer(np.outer(np.outer(a,a),a),a)
    return ans.T/(nx-bias)
ra_kurtosis.name="Raw_kurtosis"


def co_kurtosis(rand_mat,bias=0):
    ck = ra_kurtosis(rand_mat,bias)
    cm = co_variance(rand_mat,1) #UNBIASED est
    ev = ex_variance(cm)
    return ck- 3*ev
co_kurtosis.name = "co_kurtosis" 

@jit
def val_substraction(CK,CV):
    nvar = CV.shape[-1]
    for i in range(nvar):
        for j in range(nvar):
            for k in range(nvar):
                for l in range(nvar):
                    CK[i,j,k,l] = (CK[i,j,k,l] - CV[i,j]*CV[k,l] - CV[i,k]*CV[j,l] - CV[i,l]*CV[j,k])
    return CK

# def outer_Variance()
def val_kurtosis(xscaled):
    n,nvar = xscaled.shape

    CK = ra_kurtosis(xscaled).reshape(nvar,nvar,nvar,nvar)
    CV= co_variance(xscaled)
    
    CK = val_substraction(CK,CV)
    CK_m = CK.reshape(nvar, nvar*nvar*nvar)
    return CK_m
val_kurtosis.name="val_kurtosis"


## -------------------------------------------------------------------------

### Following function isued for testing accuracy with index definitio of kurtosis
### Warning-~very slow execution without numba jit


@jit
def ex_kurtosis(u):
    nx,nv = u.shape
    mom = np.zeros((nv, nv))#, dtype=float, order='F')
    # compute covariance matrix
    for j in range(nv):
        for i in range(nv):
            for n in range(nx):
                mom[i,j] = mom[i,j] + u[n,i] * u[n,j]                
    mom2 = mom/nx 

    tmp = np.zeros((nv,nv,nv,nv))#, dtype=float, order='F')
    # compute cokurtosis matrix
    for l in range(nv):
        for k in range(nv):
            for j in range(nv):
                for i in range(nv):
                    for n in range(nx):
                        tmp[i,j,k,l] = tmp[i,j,k,l] + u[n,i] * u[n,j] * u[n,k] * u[n,l]    
    
    tmp=tmp/nx
    
    for l in range(nv):
        for k in range(nv):
            for j in range(nv):
                for i in range(nv):
                    tmp[i,j,k,l] = tmp[i,j,k,l] - mom2[i,j]*mom2[k,l] - mom2[i,k]*mom2[j,l] - mom2[i,l]*mom2[j,k]
                    
    return tmp.reshape(nv,nv**3)
ex_kurtosis.name="For loop kurtosis"
