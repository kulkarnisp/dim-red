import posix
import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pathlib
import imp
from numpy.lib.nanfunctions import _nanmedian_small

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
    def __init__(self,data_name,file_name,IREQD=-4) -> None:
        data_path = os.path.join(datapath__,data_name)
        with open(os.path.join(data_path,'variable.txt'),"r") as f:
            variables = f.read()
        self.xvar = variables.split(" ")[:IREQD]
        self.xorig = np.load(os.path.join(data_path,file_name+".npy"))[:,:,:IREQD]
        self.x = self.xorig[:,:,:IREQD]
        self.xname = file_name
        self.xpath = data_path
        self.varid = {v:i for i,v in enumerate(self.xvar)}
        self.IREQD = IREQD


    def plotImg(self,species):
        x = self.x[:,:,species]
        nx,ny = x.shape
        aspect=ny/nx*0.82
        plt.imshow(x,cmap="jet",aspect=aspect)
        plt.colorbar()
        plt.title(f"Data {self.xname}- spec {self.xvar[species]}")

