import numpy as np
import dimred
from dimred.flowpipe import Elbow2

from ipywidgets import widgets
from ipywidgets import interactive,interact,interact_manual,interactive_output,interaction
from  ipywidgets.widgets import Tab,IntSlider,Dropdown,IntRangeSlider,VBox,HBox,HTML
from IPython.display import display




def Knob2(rn):
    species = rn.data.varid

    def _getTimeSlider(maxVal=10):
        return IntSlider(value=1,min=0,max=maxVal,step=1)

    def _getSpeciesMenu(options=species):
        return Dropdown(options=options,description='Select species -')
    def _getSpeciesVector(options = range(len(species))):
        return Dropdown(options=options,description='Retained vectors',value=5)

    spec_select = _getSpeciesMenu()

    out1 = widgets.interactive_output(rn.mf_plot,{'spec':spec_select})
    datatab = VBox(children=[
        spec_select,
        out1
    ],title='Data')

    out2 = widgets.interactive_output(rn.mf_embed,{'n_retain':_getTimeSlider()})
    emb_descr= widgets.Text(
                    value='Low dimensional embedding of original HCCI dataset',
                    placeholder='3D projection scatterplot',
                    description='Effect:',
                    disabled=False)

    embedtab= VBox(children=[ emb_descr,out2])
    
    retain_slider = _getSpeciesVector()
    out3 = widgets.interactive_output(rn.mf_build,{'n_retain':retain_slider}) 
    specs = _getSpeciesMenu()
    out4 = widgets.interactive_output(rn.mf_compare,{'specs':specs})
    horz = Dropdown(options=[True,False],description='Horizantal- ')
    out5 =  widgets.interactive_output(rn.mf_errors,{'n_retain':retain_slider,'horz':horz})

    midList = [retain_slider,out3]
    inigram = VBox(children=midList +[specs,out4])
    errgram = VBox(children=midList+ [horz,out5])

    scales = Dropdown(options=['linear','log'],description='Scale')
    out6 = interactive_output(rn.mf_orient,{'scale':scales})
    orientab = VBox(children=[scales,out6])
    
    specs2 = Dropdown(options=species)
    xerox =VBox(children=[specs2])

    topbar = Tab(children=[datatab,embedtab,inigram,errgram,orientab,xerox])
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