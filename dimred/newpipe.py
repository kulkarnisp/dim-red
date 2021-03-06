import imp
import numpy as np
import os,sys
import pickle

from pandas.io.formats.format import SeriesFormatter
import cantera as ct
import matplotlib.pyplot as pltss
import matplotlib.pyplot as plt

from dimred.data.loader import LoadMPI, LoadNumpy, LoadOne
from dimred.data.preprocess import AvgMaxScalar,  MinMaxScalar, Shaper, MaxAvgScalar
# dimred.
from dimred.data.preprocess import scale_sanity,Scalar

from dimred.models.linear.transform import Kurtosis
from dimred.models.linear.transform import co_variance,co_kurtosis #,outer co kurtosis
from dimred.models.linear.transform import val_kurtosis,ra_kurtosis
from dimred.tester.plotting import plot_embedding,plot_compare,plot_spectra,plot_bars,img_compare, three_compare
from dimred.tester.metrics import mean_sq_error,mean_abs_error,abs_err

from tqdm.notebook import tqdm_notebook as tqdm
# from tqdm import tqdm
from IPython.display import clear_output

wh_kurtosis = val_kurtosis
## define which kurtosis to use for calculation


def reshape_step(xinput):
    nvs = xinput.shape[-1]  ## reshapeing to oned array for cantera
    xinp = xinput.reshape(-1,nvs)
    return xinp



def transform_step(xinput,retain=4,plots=True,verbose=2,moment=wh_kurtosis,scalar=AvgMaxScalar,plt_ax=None):
    xold = xinput
    ## Reading and scaling data:-->
    xrig = xinput.copy()

    spr = Shaper()
    xrig = spr.fit_transform(xrig)
    if verbose>0:
        print("Orignial data")
        scale_sanity(xrig)
    slr = Scalar(scalar)
    xscaled = slr.fit_transform(xrig) #scale_sanity(xscaled)
    
    ## Moment calculation: -->
    Kurtosis()
    clf = Kurtosis(moment=moment,n_retain=retain)
    xred = clf.fit_transform(xscaled)
    
    ## linear reconstruction:-->
    xpew = clf.transform2(xred)
    ## xred is x reduced| et voila
    xnew = slr.transform2(xpew)
    if verbose>0:
        print("Reconstructed data")
        scale_sanity(xnew)
    xnew = spr.transform2(xnew)
    err = mean_sq_error(xnew,xold)
    clf.err = err
    ## plotting and results: -->
#     if plots:
#         plot_compare(xold,xnew)
    if plots:
        plot_embedding(xred,titler=f"{moment.name}_space-{err:.3e}",
                        color_spec=np.sum(xscaled,axis=1),cmap="jet",ax=plt_ax)    
    if verbose>=-1:
        print(f"{moment.name} reconstruction error after retaining {retain} vectors is {err:.3e}")
    return clf,xold,xnew


## CAntera--------------------------------------------------------------------------------->

# syngas = ct.Solution('cantera-module/COH2.cti')
# syngas = ct.Solution('cantera-module/premix.cti')

indices = {'temp':-2,'press':-1}
references = {'press':1.41837E+05,'temp':120} ## same keys
references = {'press':1.,'temp':1.} ## same keys


def cantera_step(xinput,gas,indices=indices,references=references,verbose=2):
# def cantera_step(xinput,gasinfo,verbose=2):
    '''# MFID is Mass fraction index in data; Samul L Jacksons MF
        # NVR is Num of variables to retain; ignore velocity components
    '''
    # gasinfo = LoadNumpy()
    # gaspath = os.path.join(gasinfo.xpath,gasinfo.chemistry)
    # gas = ct.Solution(gaspath)
    MFID=-2
    xinp = reshape_step(xinput) #.copy()       
    nx,nvs = xinp.shape

    ref_array = np.ones(nvs) #reference values required for conversion into the SI units
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
        gas.TP = sample[MFID:]                             #setting up the temperature and pressure of gas
        prod_rates.append(gas.net_production_rates)      #calculating production/consumption rates
        react_rates.append(gas.net_rates_of_progress)    #calculating reaction rates
        heat_rates.append(gas.heat_release_rate)    #calculating reaction rates

    retval = {'production':np.array(prod_rates),'reaction':np.array(react_rates),
            'mass':xinput,'hrr':np.array(heat_rates).reshape(-1,1)}
    return retval



def build_dictionary(xinput,retain=4,scalar=AvgMaxScalar,gasobj=None):
    xrig = xinput
    total = {}
    # loader.plotLine(spec=12,time=time_step)
    cmv,xold,xnew = transform_step(xrig,retain=retain,moment=co_variance, scalar=scalar,verbose=0,plots=False)
    news = cantera_step(xnew,verbose=0,gas=gasobj) 
    olds = cantera_step(xold,verbose=0,gas=gasobj)
    total['covariance'] = {'old':olds,'new':news}

    cmk,xold,xnew = transform_step(xrig,retain=retain,moment=wh_kurtosis, scalar=scalar,verbose=0,plots=False)
    news = cantera_step(xnew,verbose=0,gas=gasobj) 
    olds = cantera_step(xold,verbose=0,gas=gasobj)
    total['cokurtosis'] = {'old':olds,'new':news}

    return total


def retain_analysis(xinput,moment=co_variance,scalar=AvgMaxScalar,retain_max=13,yscale='linear',err_criterion=mean_sq_error):
    xrig = reshape_step(xinput).copy()    
    slr = Scalar(scalar)
    slr.fit(xrig)
    xscaled = slr.transform(xrig) #scale_sanity(xscaled)
    
    ## Moment calculation: -->
    clf = Kurtosis(n_retain=3,moment=moment)
    clf.fit(xscaled)  ## or co_variance; user inputx
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
    def __init__(self,data_name='premixOne.npy') -> None:
        # self.loader = LoadOne(gas)
        self.loader = LoadNumpy(data_name[:4],data_name)
        self.gas = data_name
        self.gasobj = self.loader.gasobj
        self.MyScalar = AvgMaxScalar
        self.IMAX = self.loader.x.shape[0]
        self.dat = self.loader.x
        self.n_retain= 4
        # pass
        # self.mf_plot(100)

    def mf_plot(self,time_step,spec=12):         
        self.loader.plotLine(self.mf_data(time_step),spec=spec)
        self.loader.plotImg(species=spec,cmap='jet')

    def mf_data(self,time_step=10,plot=False):
        # return self.loader.getTime(time_step,verbose=-1)[:,:-3]
        return self.loader.x[time_step]

    def mf_retain(self,time_step,scale='log'):
        # xrig = loader.getTime(time_step,verbose=-1)[:,:14]
        xrig = self.mf_data(time_step)
        verr = retain_analysis(xrig,moment=co_variance,scalar=self.MyScalar)
        kerr = retain_analysis(xrig,moment=wh_kurtosis,scalar=self.MyScalar)
        fig = plot_spectra(verr,kerr,0,0,(verr-kerr),ylabels=["Reconstruction error","Difference in error"],scale=scale)
        # fig = plot_compare(verr,kerr,titler="Moment comparison",species=0,labels=["Variance","Kurtosis"])
        # fig.axes[0].set_xlabel("Number of retained vectors")
        # fig.axes[0].set_ylabel("Species reconstruction error")
        # fig.axes[0].set_yscale(scale)

    def mf_embed(self,time_step=10):
        xrig = self.mf_data(time_step)
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(121, projection='3d')
        cmv,xold,xnew = transform_step(xrig,moment=co_variance, scalar=self.MyScalar,plt_ax=ax,verbose=False)
        self.cmv = cmv
        ax = fig.add_subplot(122, projection='3d')
        cmk,kold,knew = transform_step(xrig,moment=wh_kurtosis, scalar=self.MyScalar,plt_ax=ax,verbose=False)        
        self.cmk = cmk

    def mf_orient(self,time_step=10):
        # time_step = 100
        xrig = self.mf_data(time_step)
        cmv,xold,xnew = transform_step(xrig,moment=co_variance, scalar=self.MyScalar,verbose=-2,plots=False)
        cmk,kold,knew = transform_step(xrig,moment=wh_kurtosis, scalar=self.MyScalar,verbose=-2,plots=False)
        plot_spectra(cmv.s,cmk.s,cmv.u,cmk.u)
    
    def mf_heatmap(self):
        slr = Scalar()
        outmat = []
        for i in range(self.IMAX):
            xrig = slr.fit_transform(self.mf_data(i))
            u1,s1,v = np.linalg.svd(co_variance(xrig).T)
            u2,s2,v = np.linalg.svd(wh_kurtosis(xrig).T,full_matrices=False)
            outmat.append(u1@u2)

    def mf_compare(self,source,specs=0,**kwargs):
        total = self.total
        x0 = total['covariance']['old'][source][:,specs]
        x1 = total['covariance']['new'][source][:,specs]
        x2 = total['cokurtosis']['new'][source][:,specs]
        # plot_compare(total[moment]['old'][source],total[moment]['new'][source],species=specs)
        three_compare(x0,x1,x2,titler=f"Species -x{self.loader.xvar[specs]}")

    def mf_build(self,time_index,n_retain=4,saver=''):
        xrig = self.mf_data(time_index)
        self.total = build_dictionary(xrig,retain=n_retain,scalar=self.MyScalar,gasobj=self.gasobj)
        self.fpickle_name = f"{saver}-errs-{self.gas}-{wh_kurtosis.name}-{self.MyScalar.name}.pkl"
        if os.path.exists(self.fpickle_name):
            with open(self.fpickle_name,'rb') as f:
                self.err_dict = pickle.load(f)
        else:
            self.mf_alldata(saver=saver)
        # return self.total

    def mf_errors(self,source,**kwargs):
        moment = "covariance"
        total = self.total
        errcv = abs_err(total[moment]['old'][source] ,total[moment]['new'][source])
        moment = "cokurtosis"
        errck = abs_err(total[moment]['old'][source] ,total[moment]['new'][source])

        plot_bars(errcv/errcv,errck/errcv,horz=False,indices=self.loader.varid.keys())
    
    def mf_alldata(self,saver=''):
        total = self.total
        moments = total.keys()
        sources = total['covariance']['old'].keys()
        err_dict = {}

        for m in moments:
            err_dict[m] = {k:[] for k in sources}
        for i in tqdm(range(self.IMAX)):
            xrig = self.mf_data(i)
            total = build_dictionary(xrig,retain=self.n_retain,scalar=self.MyScalar,gasobj=self.gasobj)
            for m in moments:
                for k in sources:
                    errik = abs_err(total[m]['old'][k] ,total[m]['new'][k])
                    err_dict[m][k].append(errik)
            clear_output(wait=True)
        for m in moments:
            for k in sources:
                err_dict[m][k] = np.array(err_dict[m][k])
        self.err_dict = err_dict
        with open(self.fpickle_name,'wb') as f:
            pickle.dump(err_dict,f)
        # return err_dict

    def mf_allerror(self,source,spec=0):
        ed = self.err_dict
        errcv = ed['covariance'][source]
        errck = ed['cokurtosis'][source]
        # plot_bars(errcv,errck,horz=False)
        plot_compare(errcv,errck,labels=["Covariance","Cokurtosis"],species=spec)

class ExtensionT(Elbow):
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


class Crossings(Elbow):
    def __init__(self) -> None:
        super().__init__()
        self.loader = LoadMPI()

    def mf_plot(self, time_step, spec=12):
        return self.loader.plotImage(time_step,spec)