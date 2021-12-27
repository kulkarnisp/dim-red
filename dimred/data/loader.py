# import posix
import numpy as np
import os,sys
import matplotlib.pyplot as plt
# import pathlib
# import imp
from numpy.lib.nanfunctions import _nanmedian_small
import cantera as ct


import pandas as pd
from .oldloader1 import LoadMPI
from .oldloader2 import LoadOne

# import plotly.express as px

# ipath__ = pathlib.Path(__file__) #.parent.resolve()
# ipath__= os.path.normpath(ipath__).split(os.path.sep)
# datapath__ = os.path.join(*ipath__[:-3],"datasets/")

datapath__ = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..', 'datasets'))

# from utils import normalise, featurise


# IREQD = -4

class LoadNumpy:
    def __init__(self,data_name,file_name,pindex=1) -> None:
        data_path = os.path.join(datapath__,data_name)
        with open(os.path.join(data_path,'variable.txt'),"r") as f:
            variables = f.read()
        self.xpath = data_path
        self.getCantera()

        self.xvar = variables.split(" ")
        IREQD = self.xvar.index('P')+pindex
        self.varid = {v:i for i,v in enumerate(self.xvar[:IREQD])}

        self.xorig = np.load(os.path.join(data_path,file_name+".npy"))
        self.x = self.xorig[:,:,:IREQD]
        self.xname = file_name
        self.IREQD = IREQD

    def getCantera(self):
        self.chemistry = [f for f in os.listdir(self.xpath) if f.endswith('.cti')]
        if len(self.chemistry)>0:
            gaspath = os.path.join(self.xpath,self.chemistry[0])
            self.gasobj =  ct.Solution(gaspath)


    def plotImg(self,species,cmap='jet'):
        x = self.x[:,:,species]
        nx,ny = x.shape
        aspect=ny/nx*0.82
        plt.imshow(x,cmap=cmap,aspect=aspect)
        plt.colorbar()
        plt.title(f"Data {self.xname}- spec {self.xvar[species]}")


    
    def plotLine(self,x,spec=3):
        # self.selectFile(time)
        plt.plot(x[:,spec],"ok")
        plt.title(f"Line plot of variable x")
        plt.xlabel("Grid location")
        plt.ylabel(f"Species {self.xvar[spec]}")
        plt.show()
