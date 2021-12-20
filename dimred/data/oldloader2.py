import numpy as np
import os,sys
import matplotlib.pyplot as plt
import imp

import pandas as pd

# import plotly.express as px

# ipath__ = pathlib.Path(__file__) #.parent.resolve()
# ipath__= os.path.normpath(ipath__).split(os.path.sep)
# datapath__ = os.path.join(*ipath__[:-3],"datasets/")

datapath__ = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..', 'datasets'))

# from utils import normalise, featurise

class LoadOne():
    def __init__(self,data_name="methane/") -> None:
        data_path = os.path.join(datapath__,data_name)
        self.data_path = data_path
        self.loadFiles()

    def loadFiles(self):
        flist = [f for f in os.listdir(self.data_path) if f.endswith(".mpi")]
        self.info = imp.load_source('variables',os.path.join(self.data_path ,'variable.txt'))
        idx = int(self.info.fnameoffset)
        self.flist = sorted(flist,key=lambda f: float(f[idx:idx+10]))
       
        variables = self.info.specs.split(',')
        self.idvar = {i:v for i,v in enumerate(variables)}
        self.varid = {v:i for i,v in enumerate(variables)}

        self.nv = len(variables)
        self.ix = 0
        self.it = 0

    def selectFile(self,time=2):
        self.it = time
        self.filer = self.flist[time]


    def readFile(self,filer=None):
        if filer==None:
            filer = self.filer
        a = np.fromfile(os.path.join(self.data_path,filer),dtype="float64")
        dat = a.reshape((self.info.nx,self.info.ny,self.info.nv),order="F").squeeze()
        return dat


    def plotLine(self,spec=3,time=2):
        self.selectFile(time)
        dat = self.readFile()
        plt.plot(dat[:,spec])

        idx = int(self.info.fnameoffset)
        plt.title(f"Time- {self.filer[idx:idx+10]}")
        plt.xlabel("Grid location")
        plt.ylabel(f"Species {self.idvar[spec]}")

    def plotImg(self,spec=2,cmap='viridis',aspect=0):
        mat = self.data[:,:,spec]
        if aspect==0:
            nx,ny = mat.shape
            aspect=ny/nx*0.82
        plt.imshow(mat,aspect=aspect,cmap=cmap)
        plt.xlabel("X-dimension")
        plt.ylabel("t-dimension")
        plt.title(f"Species {self.idvar[spec]}")
        plt.colorbar()


    def getData(self):
        temp_arr = []
        for f in self.flist:
            temp_arr.append(self.readFile(f))
        self.data= np.array(temp_arr)
        return self.data

    def getDomain(self,n_domains=(1,4)):
        dat = self.getData()
        nt,nx,nv = dat.shape
        dt,dx = n_domains
        vx = int(nx/dx)
        vt = int(nt/dt)
        retlist = []
        for j in range(dt):
            retlist.extend([dat[j*vt:vt*(j+1),vx*i:vx*(i+1),:] for i in range(dx)])
        return retlist
       
    def getTime(self,time=100,verbose=0):
        fname = self.flist[time]
        if verbose:
            print(f"reading {time}th file {fname}...")    
        dat = self.readFile(fname)
        # df = pd.DataFrame(dat)
        # df.columns = self.idvar.values()
        return dat #df


