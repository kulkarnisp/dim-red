import posix
import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pathlib
import imp


# import plotly.express as px


# from utils import normalise, featurise

class LoadOne():
    def __init__(self,data_path="../datasets/methane/") -> None:
        self.data_path = data_path
        self.loadFiles()

    def loadFiles(self):
        flist = [f for f in os.listdir(self.data_path) if f.endswith(".mpi")]
        self.info = imp.load_source('variables',self.data_path + 'variables.txt')
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
        a = np.fromfile(self.data_path+ filer,dtype="float64")
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
            print(f"reading file {fname}...")    
        return self.readFile(fname)






class LoadMPI():
    def __init__(self,fpath=0):
        self.inipath = "/home/shubham/strial/Combust/hcci/data/"
        if fpath==0:
            self.fpath = self.inipath
            self.loadFile("hcci")
        else:
            self.fpath = fpath
            self.fpath = self.fpath.replace("hcci","isml")
        
        
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
    
    def getHrr(self):
        n = self.file.find('E')
        root = self.file[n-6:n+4]
        a = np.fromfile(self.fpath[:-5]+"hrr/hrr."+root+".mpi",
                        count=self.nx*self.ny*1,dtype="float64")
        dat = a.reshape((self.nx,self.ny,1), order="F") 
        return dat      
    
    def getMat(self,species = 2):
#         i = self.variables[t]
        self.filer.seek(self.i*8*self.nx*self.ny+self.badOffset)
        a = np.fromfile(self.filer,
                        count=self.nx*self.ny,dtype="float64")
        T = a.reshape(self.nx,self.ny,order="F")        
        return T

    def getTrain(self, j):
        self.file = self.flist[j]
        y = self.getHrr()
        x = self. getDat(j)
        return x,y
    
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

    

class LoadOned(LoadMPI):
    def __init__(self, fpath=0):
        super().__init__(fpath=fpath)