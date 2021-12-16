import numpy as np
import os,sys
import pickle

from pandas.io.formats.format import SeriesFormatter
import cantera as ct
import matplotlib.pyplot as plt

from dimred.data.loader import LoadOne
from dimred.data.preprocess import AvgMaxScalar, MinMaxScalar, Shaper, MaxAvgScalar
# dimred.
from dimred.data.preprocess import scale_sanity,Scalar

from dimred.models.linear.transform import Kurtosis
from dimred.models.linear.transform import co_variance #,co_kurtosis
from dimred.models.linear.transform import val_kurtosis as co_kurtosis
# from dimred.models.linear.transform import ra_kurtosis as co_kurtosis
from dimred.tester.plotting import plot_embedding,plot_compare,plot_spectra,plot_bars,img_compare
from dimred.tester.metrics import mean_sq_error,mean_abs_error,abs_err

def reshape_step(xinput):
    nvs = xinput.shape[-1]  ## reshapeing to oned array for cantera
    xinp = xinput.reshape(-1,nvs)
    return xinp


def transform_step(xinput,retain=4,plots=True,verbose=2,moment=co_kurtosis,scalar=AvgMaxScalar(),plt_ax=None):
    xrig = reshape_step(xinput).copy()    
    ## Reading and scaling data:-->
    xold = xrig.copy()
    if verbose>0:
        print("Orignial data")
        scale_sanity(xrig)
    slr = Scalar(scalar)
    slr.fit(xrig)
    xscaled = slr.transform(xrig) #scale_sanity(xscaled)
    
    ## Moment calculation: -->
    clf = Kurtosis(n_retain=retain)
    clf.fit(xscaled,moment=moment)  ## or co_variance; user inputx
    xred = clf.transform(xscaled)
    
    ## linear reconstruction:-->
    xpew = clf.transform2(xred)
    ## xred is x reduced| et voila
    xnew = slr.transform2(xpew)
    err = mean_sq_error(xnew,xold)
    clf.err = err
    ## plotting and results: -->
#     if plots:
#         plot_compare(xold,xnew)
    if plots:
        plot_embedding(xred,titler=f"{moment.name}_space-{err:.3e}",
                        color_spec=np.sum(xscaled,axis=1),cmap="jet",ax=plt_ax)    
    if verbose>0:
        print("Reconstructed data")
        scale_sanity(xnew)
    if verbose>=-1:
        print(f"{moment.name} reconstruction error after retaining {retain} vectors is {err:.3e}")
    return clf,xold,xnew


## CAntera--------------------------------------------------------------------------------->

syngas = ct.Solution('cantera-module/COH2.cti')

indices = {'temp':12,'press':13}
references = {'press':1.41837E+05,'temp':120} ## same keys


def cantera_step(xinput,NVR=14,MFID=12,indices=indices,references=references,gas=syngas,verbose=2):
    '''# MFID is Mass fraction index in data; Samul L Jacksons MF
        # NVR is Num of variables to retain; ignore velocity components
    '''
    xinp = reshape_step(xinput).copy()       
    nx,nvs = xinp.shape
    xinp = xinp[:,:NVR]

    ref_array = np.ones(NVR) #reference values required for conversion into the SI units
    for k,v in indices.items():
        ref_array[v] = references[k]

    if verbose:   ## print gas properties
        print(gas())

    ## below are hardcoded numbers --- TBD

    prod_rates = []
    react_rates = []
    heat_rates = []
    for i in range(nx):                                  #iterating over all grid points
    #     print(i)                                            
        sample = xinp[i]*ref_array                 #converting into the SI units
        gas.Y = sample[:MFID]                                #setting up the mass fraction of the species
        gas.TP = sample[MFID:NVR]                             #setting up the temperature and pressure of gas
        prod_rates.append(gas.net_production_rates)      #calculating production/consumption rates
        react_rates.append(gas.net_rates_of_progress)    #calculating reaction rates
        heat_rates.append(gas.heat_release_rate)    #calculating reaction rates

    retval = {'production':np.array(prod_rates),'reaction':np.array(react_rates),
            'mass':xinput,'hrr':np.array(heat_rates).reshape(-1,1)}
    return retval



def build_dictionary(xinput,retain=4,scalar=AvgMaxScalar()):
    xrig = xinput
    total = {}
    # loader.plotLine(spec=12,time=time_step)
    cmv,xold,xnew = transform_step(xrig,retain=retain,moment=co_variance, scalar=scalar,verbose=0,plots=False)
    news = cantera_step(xnew,verbose=0) 
    olds = cantera_step(xold,verbose=0)
    total['covariance'] = {'old':olds,'new':news}

    cmk,xold,xnew = transform_step(xrig,retain=retain,moment=co_kurtosis, scalar=scalar,verbose=0,plots=False)
    news = cantera_step(xnew,verbose=0) 
    olds = cantera_step(xold,verbose=0)
    total['cokurtosis'] = {'old':olds,'new':news}

    return total


def retain_analysis(xinput,moment=co_variance,scalar=AvgMaxScalar(),retain_max=13,yscale='linear',err_criterion=mean_sq_error):
    xrig = reshape_step(xinput).copy()    
    slr = Scalar(scalar)
    slr.fit(xrig)
    xscaled = slr.transform(xrig) #scale_sanity(xscaled)
    
    ## Moment calculation: -->
    clf = Kurtosis(n_retain=3)
    clf.fit(xscaled,moment=moment)  ## or co_variance; user inputx
    errs = []
    for i in range(0,retain_max):
        clf.n_retain = i
        xred = clf.transform(xscaled)
        
        ## linear reconstruction:-->
        xpew = clf.transform2(xred)
        ## xred is x reduced| et voila
        xnew = slr.transform2(xpew)
        errs.append(err_criterion(xnew,xinput))
    return np.array(errs)    


## resf of the f owl

class Elbow:
    def __init__(self) -> None:
        self.loader = LoadOne()
        self.MyScalar = AvgMaxScalar()
        self.IMAX = 201
        # pass
        # self.mf_plot(100)

    def mf_plot(self,time_step,spec=12):
        self.loader.plotLine(spec=spec,time=time_step)

    def mf_data(self,time_step=100,plot=False):
        return self.loader.getTime(time_step,verbose=-1)[:,:14]

    def mf_retain(self,time_step,scale='log'):
        # xrig = loader.getTime(time_step,verbose=-1)[:,:14]
        xrig = self.mf_data(time_step)
        verr = retain_analysis(xrig,moment=co_variance,scalar=self.MyScalar)
        kerr = retain_analysis(xrig,moment=co_kurtosis,scalar=self.MyScalar)
        fig = plot_spectra(verr,kerr,0,0,(verr-kerr),ylabels=["Reconstruction error","Difference in error"],scale=scale)
        # fig = plot_compare(verr,kerr,titler="Moment comparison",species=0,labels=["Variance","Kurtosis"])
        # fig.axes[0].set_xlabel("Number of retained vectors")
        # fig.axes[0].set_ylabel("Species reconstruction error")
        # fig.axes[0].set_yscale(scale)

    def mf_embed(self,time_step=100):
        xrig = self.mf_data(time_step)
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(121, projection='3d')
        cmv,xold,xnew = transform_step(xrig,moment=co_variance, scalar=self.MyScalar,plt_ax=ax,verbose=False)
        self.cmv = cmv
        ax = fig.add_subplot(122, projection='3d')
        cmk,kold,knew = transform_step(xrig,moment=co_kurtosis, scalar=self.MyScalar,plt_ax=ax,verbose=False)        
        self.cmk = cmk

    def mf_orient(self,time_step=100):
        # time_step = 100
        xrig = self.mf_data(time_step)
        cmv,xold,xnew = transform_step(xrig,moment=co_variance, scalar=self.MyScalar,verbose=-2,plots=False)
        cmk,kold,knew = transform_step(xrig,moment=co_kurtosis, scalar=self.MyScalar,verbose=-2,plots=False)
        plot_spectra(cmv.s,cmk.s,cmv.u,cmk.u)
    
    def mf_heatmap(self):
        slr = Scalar()
        outmat = []
        for i in range(self.IMAX):
            xrig = slr.fit_transform(self.mf_data(i))
            u1,s1,v = np.linalg.svd(co_variance(xrig).T)
            u2,s2,v = np.linalg.svd(co_kurtosis(xrig).T,full_matrices=False)
            outmat.append(u1@u2)

    def mf_compare(self,moment,source,specs=0):
        total = self.total
        plot_compare(total[moment]['old'][source],total[moment]['new'][source],species=specs)

    def mf_build(self,time_index,n_retain=4):
        xrig = self.mf_data(time_index)
        self.total = build_dictionary(xrig,retain=n_retain,scalar=self.MyScalar)
        # return self.total

    def mf_errors(self,source):
        moment = "covariance"
        total = self.total
        errcv = abs_err(total[moment]['old'][source] ,total[moment]['new'][source])
        moment = "cokurtosis"
        errck = abs_err(total[moment]['old'][source] ,total[moment]['new'][source])

        plot_bars(errcv/errcv,errck/errcv,horz=False,indices=self.loader.varid.keys())
    
    def mf_alldata(self,saver=None):
        total = self.total
        moments = total.keys()
        sources = total['covariance']['old'].keys()
        err_dict = {}

        for m in moments:
            err_dict[m] = {k:[] for k in sources}
        for i in range(self.IMAX):
            self.mf_build(i,n_retain=4)
            total = self.total
            for m in moments:
                for k in sources:
                    errik = abs_err(total[m]['old'][k] ,total[m]['new'][k])
                    err_dict[m][k].append(errik)
        for m in moments:
            for k in sources:
                err_dict[m][k] = np.array(err_dict[m][k])
        self.err_dict = err_dict
        if saver!=None:
            pickle.dump(err_dict)
        # return err_dict

    def mf_allerror(self,source,spec=0):
        ed = self.err_dict
        errcv = ed['covariance'][source]
        errck = ed['cokurtosis'][source]
        # plot_bars(errcv,errck,horz=False)
        plot_compare(errcv,errck,labels=["Covariance","Cokurtosis"],species=spec)

class RunnerDomain(Elbow):
    def __init__(self, domains = (1,4)) -> None:
        super().__init__()
        xregions = [self.loader.getData()]
        self.loader.plotImg(spec=12)#,aspect=0.9)

        xregions.extend(self.loader.getDomain(domains))
        self.xregions = xregions
        self.IMAX = len(xregions)
        self.spr = Shaper()
    
    def mf_data(self, time_step=0, plot=False):
        return self.spr.fit_transform(self.xregions[time_step])

    def mf_images(self,moment,source,specs=0):
        total = self.total
        spr = self.spr
        img_compare(spr.transform2(total[moment]['old'][source]),
                 spr.transform2(total[moment]['new'][source]),species=specs)