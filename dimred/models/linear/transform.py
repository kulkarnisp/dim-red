# utilities used in kurtosis analysis

import numpy as np
import os,time
from .moments import co_variance,co_kurtosis,val_kurtosis,ra_kurtosis


class Kurtosis:
    def __init__(self,moment=co_variance,n_retain=4,mom_path=None) -> None:
        self.n_retain = n_retain
        self.mom_path = mom_path
        self.moment = moment
        self.namer = moment.name
        # pass

    def _recallMoments(self):
        ## todo add sanity checks, file exists ! is valid?
        if os.path.isfile(self.mom_path):
            self.cm = np.load(self.mom_path)
        else:
            print(f"Moment file not found, calculating {self.namer}")
            self._calcMoment()
            print(f"saving {self.namer} at {self.mom_path}")
            np.save(self.mom_path,self.cm)


    def _calcMoment(self):
        start = time.time()
        self.cm = self.moment(self.x)
        end = time.time()
        print(f"Time required for {self.namer} is {(end-start):4e} sec")

    def fit(self, X):
        self.x = X
        if self.mom_path!=None:
            self._recallMoments()
        else:
            self._calcMoment()
        # self.cm = self.moment(X)
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