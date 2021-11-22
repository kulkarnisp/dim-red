# from ..helper import *
import numpy as np
from functools import partial



class Distance():
    def __init__(self,name="kurtosis"):
        self.n =0
        self.name = name
        self.__class__.__name__ = "DistancesUS"
        
    def fit(self,X):
        self.vec = np.svd(X)
       
    def predict(self,X):
        y = 1- self.decision_function(X)
        yh = (y- y.min())/(y.max()-y.min())
        return  -2*np.array(yh >0.5,dtype=int) + 1
    
    def fit_predict(self,X):
        self.fit(X)
        return self.predict(X)
    
    def decision_function(self,X):
#         y = np.sum(X@self.vec,axis=1)
        y = X@self.vec
#         y = X@self.vec
        return y/y.max()
        
    
 