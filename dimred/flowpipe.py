import numpy as np
import os,sys
import pickle

import matplotlib.pyplot as plt

from dimred.data.loader import LoadMPI, LoadNumpy, LoadOne,datapath__
from dimred.data.preprocess import AvgMaxScalar, MinMaxScalar, Shaper, MaxAvgScalar
# dimred.
from dimred.data.preprocess import scale_sanity,Scalar

from dimred.models.linear.transform import Kurtosis
from dimred.models.linear.transform import co_variance,co_kurtosis #,outer co kurtosis
from dimred.models.linear.transform import val_kurtosis,ra_kurtosis
from dimred.tester.plotting import plot_embedding,plot_compare,plot_spectra,plot_bars,img_compare
from dimred.tester.metrics import mean_sq_error,mean_abs_error,abs_err

wh_kurtosis = val_kurtosis
mom_folder = os.path.join(datapath__,"moments")

class ServicePIpe:
    def __init__(self,data,moment=co_variance,scalar=AvgMaxScalar,n_retain=4) -> None:
        self.xinput = data.x.copy()
        self.xname = data.xname
        self.shaper= Shaper()
        self.scalar = Scalar(scalar)
        self.model = Kurtosis(moment=moment,n_retain=n_retain)
        self.pipeline = [self.shaper,self.scalar,self.model]
        ## Todo add post processing class
    
    def encode(self,xraw=None):
        global mom_folder
        data_name = self.xname
        if xraw is None:
            xraw = self.xinput.copy()
        if data_name is not None:
            fname = f"{data_name}-{self.scalar.namer}-{self.model.namer}.npy"
            mom_path = os.path.join(mom_folder,fname)
        else:
            mom_path = None
        x = self.shaper.fit_transform(xraw)
        x = self.scalar.fit_transform(x)
        self.model.mom_path = mom_path
        x = self.model.fit_transform(x)
        # for p in self.process.values():
        #     xencode = p.fit_transform(xencode)
        self.xencoded = x
        return x

    def decode(self,xencode=None):
        if xencode is None:
            xencode = self.xencoded.copy()
        x = self.model.transform2(xencode)
        x = self.scalar.transform2(x)
        self.xdecoded = self.shaper.transform2(x)
        return self.xdecoded

    def build(self):
        xold = self.xinput
        xenc = self.encode()
        xnew = self.decode()
        self.err = np.sum(np.abs(xnew-xold))
        # return {'old':xold,'new':xnew,'enc':xenc}


# def retain_analysis(server):
#     server = ServicePIpe()
#     server


# def retain_analysis(xinput,server,yscale='linear'):
#     server = ServicePIpe()
#     errs = []
#     retain_max = server.xencoded.shape[-1]
#     for i in range(0,retain_max):
#         clf.n_retain = i
#         xred = clf.transform(xscaled)
        
#         ## linear reconstruction:-->
#         xpew = clf.transform2(xred)
#         ## xred is x reduced| et voila
#         xnew = slr.transform2(xpew)
#         errs.append(mean_sq_error(xnew,xinput))
#     return np.array(errs)    

class Elbow2:
    def __init__(self,data) -> None:
        self.data = data
        self.myScalar = AvgMaxScalar
        self.mf_build(4)
        # pass

    def mf_build(self,n_retain=4):
        data = self.data
        self.total = {}
        self.vari = ServicePIpe(data,moment=co_variance,n_retain=n_retain,scalar=self.myScalar)
        self.vari.build()
        self.kurt = ServicePIpe(data,moment=wh_kurtosis,n_retain=n_retain,scalar=self.myScalar)
        self.kurt.build()

    def mf_plot(self,spec=0):
        T = self.data.x[:,:,spec]
        titler = self.data.xname +"-"+ self.data.xvar[spec] 
        plt.figure(figsize=(6,6))
        fig = plt.imshow(T,cmap = "jet",aspect=1)
        plt.title(titler)
        plt.colorbar()
        plt.show()
        

    def mf_embed(self,n_retain=0):
        fig = plt.figure(figsize=(12,5))

        for i,server in enumerate([self.vari,self.kurt]):
            ax = fig.add_subplot(121+i, projection='3d')
            xred = server.xencoded
            err = server.err
            namer = server.model.namer
            plot_embedding(xred,titler=f"{namer}_space-{err:.3e}",
                color_spec=np.sum(xred,axis=1),cmap="jet",ax=ax)    

    def mf_orient(self,scale='log'):
        # time_step = 100
        cmv = self.vari.model
        cmk = self.kurt.model
        plot_spectra(cmv.s,cmk.s,cmv.u,cmk.u,scale=scale)

    
    def mf_compare(self,specs=0):
        labels = ["Covariance-original","Cokurtosis-original"]
        x0 = self.vari.xinput
        x1 = self.vari.xdecoded
        x2 = self.kurt.xdecoded
        img_compare(x1-x0,x2-x0,species=specs,labels=labels)

    def mf_errors(self,n_retain=4,horz=True,sources='',):
        x0 = self.vari.xinput
        errcv = np.sum(abs(self.vari.xdecoded -x0),axis=(0,1))
        errck = np.sum(abs(self.kurt.xdecoded -x0),axis=(0,1))
        plot_bars(errcv/errcv,errck/errcv,horz=horz,indices=self.data.xvar)


# class MomAnalysis:
#     def __init__(self) -> None:
#         self.total = {}
#         pass

#     def embed_data(self,dat):
#         self.cvr = ServicePIpe(dat,moment=co_kurtosis)
#         self.cmk = ServicePIpe(dat,moment=co_kurtosis)


#     def build_dictionary(self,dat,retain=4,scalar=AvgMaxScalar):


# class PostProcess:
#     def __init__(self,server) -> None:
#         self.server=server
#         pass

    
    
#     def embedd_analysis(self):

    