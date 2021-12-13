import numpy as np
import os,sys

from pandas.io.formats.format import SeriesFormatter
import cantera as ct
import matplotlib.pyplot as plt

from dimred.data.loader import LoadOne
from dimred.data.preprocess import MinMaxScalar, Shaper,ZeroMeanScalar,MeanMaxScalar,AvgMaxScalar
# dimred.
from dimred.data.preprocess import scale_sanity,Scalar

from dimred.models.linear.transform import Kurtosis
from dimred.models.linear.transform import co_variance,co_kurtosis
from dimred.tester.plotting import plot_embedding,plot_compare,plot_spectra,plot_bars,img_compare
from dimred.tester.metrics import mean_sq_error,mean_abs_error,abs_err

def reshape_step(xinput):
    nvs = xinput.shape[-1]  ## reshapeing to oned array for cantera
    xinp = xinput.reshape(-1,nvs)
    return xinp


def transform_step(xinput,retain=4,plots=True,verbose=2,moment=co_kurtosis,scalar=AvgMaxScalar,plt_ax=None):
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
        print(f"{moment.name} reconstruction error after retaining {retain} vectors is {err:.4f}")
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
    for i in range(nx):                                  #iterating over all grid points
    #     print(i)                                            
        sample = xinp[i]*ref_array                 #converting into the SI units
        gas.Y = sample[:MFID]                                #setting up the mass fraction of the species
        gas.TP = sample[MFID:NVR]                             #setting up the temperature and pressure of gas
        prod_rates.append(gas.net_production_rates)      #calculating production/consumption rates
        react_rates.append(gas.net_rates_of_progress)    #calculating reaction rates
    return np.array(prod_rates),np.array(react_rates)



def build_dictionary(xinput,retain=4):
    xrig = xinput
    cmv,xold,xnew = transform_step(xrig,retain=retain,moment=co_variance, scalar=AvgMaxScalar,verbose=0,plots=False)
    cmk,kold,knew = transform_step(xrig,retain=retain,moment=co_kurtosis, scalar=AvgMaxScalar,verbose=0,plots=False)
    total = {}
    # loader.plotLine(spec=12,time=time_step)
    prod_new,react_new = cantera_step(xnew,verbose=0) 
    prod_old,react_old = cantera_step(xold,verbose=0)

    kprod_new,kreact_new = cantera_step(knew,verbose=0) 
    kprod_old,kreact_old = cantera_step(kold,verbose=0)



    olds={'production':prod_old,'reaction':react_old,'mass':xold}
    news={'production':prod_new,'reaction':react_new,'mass':xnew}

    total['covariance'] = {'old':olds,'new':news}

    olds={'production':kprod_old,'reaction':kreact_old,'mass':kold}
    news={'production':kprod_new,'reaction':kreact_new,'mass':knew}
    total['cokurtosis'] = {'old':olds,'new':news}
    return total


def retain_analysis(xinput,moment=co_variance,scalar=AvgMaxScalar,retain_max=13,yscale='linear',err_criterion=mean_sq_error):
    xrig = reshape_step(xinput).copy()    
    slr = Scalar(scalar)
    slr.fit(xrig)
    xscaled = slr.transform(xrig) #scale_sanity(xscaled)
    
    ## Moment calculation: -->
    clf = Kurtosis(n_retain=3)
    clf.fit(xscaled,moment=moment)  ## or co_variance; user inputx
    errs = []
    for i in range(1,retain_max):
        clf.n_retain = i
        xred = clf.transform(xscaled)
        
        ## linear reconstruction:-->
        xpew = clf.transform2(xred)
        ## xred is x reduced| et voila
        xnew = slr.transform2(xpew)
        errs.append(err_criterion(xnew,xinput))
    return np.array(errs)    


## resf of the f owl

class RunnerTime:
    def __init__(self) -> None:
        self.loader = LoadOne()
        # pass
        # self.mf_plot(100)

    def mf_plot(self,time_step,spec=12):
        self.loader.plotLine(spec=spec,time=time_step)

    def mf_data(self,time_step=100,plot=False):
        return self.loader.getTime(time_step,verbose=-1)[:,:14]

    def mf_retain(self,time_step,scale='log'):
        # xrig = loader.getTime(time_step,verbose=-1)[:,:14]
        xrig = self.mf_data(time_step)
        verr = retain_analysis(xrig,moment=co_variance).reshape(-1,1)
        kerr = retain_analysis(xrig,moment=co_kurtosis).reshape(-1,1)
        fig = plot_compare(verr,kerr,titler="Moment comparison",species=0,labels=["Kurtosis","Variance"])
        fig.axes[0].set_xlabel("Number of retained vectors")
        fig.axes[0].set_ylabel("Species reconstruction error")
        fig.axes[0].set_yscale(scale)

    def mf_embed(self,time_step=100):
        xrig = self.mf_data(time_step)
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(121, projection='3d')
        cmv,xold,xnew = transform_step(xrig,moment=co_variance, scalar=AvgMaxScalar,plt_ax=ax,verbose=False)

        ax = fig.add_subplot(122, projection='3d')
        cmk,kold,knew = transform_step(xrig,moment=co_kurtosis, scalar=AvgMaxScalar,plt_ax=ax,verbose=False)        

    def mf_orient(self,time_step=100):
        # time_step = 100
        xrig = self.mf_data(time_step)
        cmv,xold,xnew = transform_step(xrig,moment=co_variance, scalar=AvgMaxScalar,verbose=-2,plots=False)
        cmk,kold,knew = transform_step(xrig,moment=co_kurtosis, scalar=AvgMaxScalar,verbose=-2,plots=False)
        plot_spectra(cmv.s,cmk.s,cmv.u,cmk.u)

    def mf_compare(self,moment,source,specs=0):
        total = self.total
        plot_compare(total[moment]['old'][source],total[moment]['new'][source],species=specs)

    def mf_build(self,time_index,n_retain=4):
        xrig = self.mf_data(time_index)
        self.total = build_dictionary(xrig,retain=n_retain)
        # return self.total

    def mf_errors(self,source):
        moment = "covariance"
        total = self.total
        errcv = abs_err(total[moment]['old'][source] ,total[moment]['new'][source])
        moment = "cokurtosis"
        errck = abs_err(total[moment]['old'][source] ,total[moment]['new'][source])

        plot_bars(errcv/errcv,errck/errcv,horz=False,indices=self.loader.varid.keys())


class RunnerDomain(RunnerTime):
    def __init__(self, domains = (1,4)) -> None:
        super().__init__()
        xregions = [self.loader.getData()]
        self.loader.plotImg(spec=12)#,aspect=0.9)

        xregions.extend(self.loader.getDomain(domains))
        self.xregions = xregions
        self.spr = Shaper()
    
    def mf_data(self, time_step=0, plot=False):
        return self.spr.fit_transform(self.xregions[time_step])

    def mf_images(self,moment,source,specs=0):
        total = self.total
        spr = self.spr
        img_compare(spr.transform2(total[moment]['old'][source]),
                 spr.transform2(total[moment]['new'][source]),species=specs)