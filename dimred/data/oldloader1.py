import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pathlib

import pandas as pd

# import plotly.express as px

# ipath__ = pathlib.Path(__file__) #.parent.resolve()
# ipath__= os.path.normpath(ipath__).split(os.path.sep)
# datapath__ = os.path.join(*ipath__[:-3],"datasets/")

datapath__ = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..', 'datasets'))

# from utils import normalise, featurise


class LoadMPI():
    def __init__(self,data='hcci'):
        self.inipath = "/home/shubham/strial/Combust/hcci/data/"
        self.fpath = self.inipath.replace("hcci",data)
        self.loadFile(data)
        
    def loadFile(self, val="hcci"):
        flist = [f for f in os.listdir(self.fpath) if f.endswith(".mpi")]
        
        self.flist = sorted(flist, key=lambda f: float(f[-20:-10]) )
        
        self.selTime(0)        
        self.loadOptions()
        self.loadVariables()

    def loadOptions(self):  
        if "hcci" in self.fpath:
            self.nx = 672
            self.ny = 672
            self.badOffset = 0
            self.species = "varhcci.txt"
        else:
            self.nx = 3600
            self.ny = 1800
            self.badOffset = 89266
            self.species = "varisml.txt"
                
    def loadVariables(self,ipath=None):
        if ipath is None:
            ipath = pathlib.Path(__file__).parent.resolve()
#        ipath = "/home/shubham/Projects/Anomaly/src/unsup/"
        fr = open(os.path.join(ipath,self.species),"r")
        ychem = fr.read()
#         jack = raw.split()
#         sparo = [v for i,v in enumerate(jack) if (i+1)%3==0]
#         ychem = " ".join(sparo)
        ychem += " T P u v w"
        variables = ychem.split(" ")

        self.idvar = {i:v for i,v in enumerate(variables)}
        self.varid = {v:i for i,v in enumerate(variables)}
        self.variables = self.varid
        self.nv = len(variables)
        self.i = 0

    def selTime(self,index):
        ## path remains same just the file changes
        self.fid = index
        self.file = self.flist[index]
        self.filer = open(self.fpath+self.file,"rb")

    def selSpec(self,index):
        ## file remains same, just index changes
        self.i = index
    
    def getDat(self,time):
        ### be careful with this function ; memory sensitive
        self.selTime(time)
        a = np.fromfile(self.filer,
                        count=self.nx*self.ny*self.nv,dtype="float64")
        dat = a.reshape((self.nx,self.ny,self.nv),order="F") 
        return dat 
    
    def getTime(self,time):
        return self.getDat(time)

    def getHrr(self):
        n = self.file.find('E')
        root = self.file[n-6:n+4]
        a = np.fromfile(self.fpath[:-5]+"hrr/hrr."+root+".mpi",
                        count=self.nx*self.ny*1,dtype="float64")
        dat = a.reshape((self.nx,self.ny,1), order="F") 
        return dat      
    
    def getMat(self,species = None):
        if species is None:
            species = self.i
#         i = self.variables[t]
        self.filer.seek(species*8*self.nx*self.ny+self.badOffset)
        a = np.fromfile(self.filer,
                        count=self.nx*self.ny,dtype="float64")
        T = a.reshape(self.nx,self.ny,order="F")        
        return T

    def getTrain(self, j):
        self.file = self.flist[j]
        y = self.getHrr()
        x = self. getDat(j)
        return x,y
    
    def plotImage(self,time=10,species=0):
        self.selTime(time)
        self.selSpec(species)
        T = self.getMat(self.i)
        # plt.figure(figsize=(6,6))
        fig = plt.imshow(T,cmap = "jet",aspect=1)
        titler = self.idvar[self.i] + f" at T{self.filer}"
        plt.title(self.file)
        plt.colorbar()
    
    def stPlot(self,saver="temp.png",titler=None,species=0):
        T = self.getMat(self.i)
        plt.figure(figsize=(6,6))
        fig = plt.imshow(T,cmap = "jet",aspect=1)
        if titler is None:
            titler = self.idvar[self.i] + f" at T{self.filer}"
        plt.title(self.file)
        plt.colorbar()
        if saver!=0:
            plt.savefig(saver)
        plt.show()
        return fig
        
    def dyPlot(self,px=plt):
        T = self.getMat()
        fig = px.imshow(T,
                        color_continuous_scale="jet",
                       title=self.file)
        fig.update_layout(
           autosize=False,
           width=800,
           height=800,)
        return fig
