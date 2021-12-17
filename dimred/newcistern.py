import numpy as np
import dimred
from dimred.newpipe import Elbow

from ipywidgets import widgets
from ipywidgets import interactive,interact,interact_manual,interactive_output,interaction
from  ipywidgets.widgets import Tab,IntSlider,Dropdown,IntRangeSlider,VBox,HBox,HTML
from IPython.display import display


def Knob(rn):
    species = rn.loader.varid
    flist  = rn.loader.flist
    nflist = len(flist)
    def _getTimeSlider(maxVal=nflist):
        return IntSlider(value=100,min=0,max=maxVal,step=1)
    def _getFileList(flist=flist):
        fdics = {f:i for f,i in enumerate(flist)}
        return Dropdown(options=fdics)
    def _getSpeciesMenu(options=species):
        return Dropdown(options=options)
    def _getSpeciesVector(options = range(len(species))):
        return Dropdown(options=options)

    time_slier = _getTimeSlider()
    time_slier2 = _getTimeSlider()
    spec_select = _getSpeciesMenu()

    out1 = widgets.interactive_output(rn.mf_plot,{'time_step':time_slier,'spec':spec_select})
    out2 = widgets.interactive_output(rn.mf_embed,{'time_step':time_slier2})

    retain_slider = _getSpeciesVector()
    time_slider3 = _getTimeSlider()
    out3 = widgets.interactive_output(rn.mf_build,{"time_index":time_slider3,'n_retain':retain_slider}) 


    moms = Dropdown(options= rn.total.keys())
    sources = Dropdown(options=rn.total['covariance']['old'].keys())
    specs = _getSpeciesMenu()
    out4 = widgets.interactive_output(rn.mf_compare,{'moment':moms,'source':sources,'specs':specs})
    out5 =  widgets.interactive_output(rn.mf_errors,{'source':sources})

    slicer = _getTimeSlider()
    out6 = interactive_output(rn.mf_orient,{'time_step':slicer})
    scales = Dropdown(options=['linear','log'])
    out7 = interactive_output(rn.mf_retain,{'time_step':slicer,'scale':scales})

    def _auto_errors(values):
        sources.set_state(sources.get_state())

    time_slider3.observe(_auto_errors)
        
    datatab = VBox(children=[
        time_slier,spec_select,
        out1
    ],title='Data')
    embedtab= VBox(children=[
        time_slier2, out2])

    midList = [time_slider3,retain_slider,out3,moms,sources,specs]
    inigram = VBox(children=midList +[out4])
    errgram = VBox(children=midList[:-3]+ [sources,out5])

    orient = VBox(children=[slicer,scales,out6,out7])
    
    sources2 = Dropdown(options=rn.total['covariance']['old'].keys())
    specs2 = Dropdown(options=species)
    out9 = interactive_output(rn.mf_allerror,{'source':sources2,'spec':specs2})

    retain =VBox(children=[sources2,specs2,out9])
    topbar = Tab(children=[datatab,embedtab,inigram,errgram,orient,retain])
    titles = ['dataset','embedding','reconstr','species','vectors','xerox']

    for i,name in enumerate(titles):
        topbar.set_title(i,name)

    return topbar



# datae = interactive(rn.mf_plot,time_step=range(201),spec=rn.loader.varid)
# embed = interactive(rn.mf_embed,time_step=range(201))
# inigram =interactive(rn.mf_build,time_index=range(201),n_retain=range(12))
# textile = interactive(rn.mf_compare,moment=rn.total.keys(),source=rn.total['covariance']['old'].keys(),specs=rn.loader.varid)
# recon = VBox(children=[inigram,textile])
# erons =  interactive(rn.mf_errors,source=rn.total['covariance']['old'].keys())

# tablist = [datae,embed,recon,erons]
# topbar = widgets.Tab(children=tablist)