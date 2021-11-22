
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import seaborn as sns

import data
import utils
import model
import test

def imax(X,ax,cmap="Reds",aspect=1.0,titler=""):
    im = ax.imshow(X,cmap=cmap,aspect=aspect)
#     plt.colorbar(im)
#    fig.colorbar(im)
    ax.grid(None)
    ax.set_title(titler)
    ax.set_xticks([])
    ax.set_yticks([])
#     plt.close()
    return ax

time = 0
ns = 6
loader = data.LoadMPI() 
loader.loadFile('hcci')
xi,yi = loader.getTrain(time)
nx,ny,nv = xi.shape

pool = 12
xr = utils.preporcess(xi,pool=pool)
yr = utils.preporcess(np.abs(yi),
                      pool=pool,
                       scalar=utils.MinMaxScalar)

ys = yr>0.5
xs = xr[:,:,:5].sum(axis=2)

fig,axe = plt.subplots(2,3,figsize=(15,6))
plt.subplots_adjust(left=0.25, bottom=0.25)
fig.suptitle("HCCI Kernels -CFPL")
imax(xi[:,:,6],axe[0,0],cmap='jet')
imax(yi[:,:,0],axe[1,0],cmap='seismic')
imax(xi[:,:,0],axe[0,1])
imax(yr,axe[1,1],cmap="Greens")
imax(ys,axe[1,2],cmap="Greens")
imax(xs,axe[0,2])

cmap = "jet"

axcolor = 'lightgoldenrodyellow'
#axcolor = 'azure'


axfreq = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)


sfreq = Slider(axfreq, 'TimeStep', 0, len(loader.flist)-1, valinit=1, valstep=1)
samp = Slider(axamp, 'Threshold', 0, 1, valinit=0,valstep =0.1)
def update(val):
    global xi,yi,ns
    ti = int(sfreq.val) ##time step
    xi,yi = loader.getTrain(ti)
#    n = np.random.randint(nv-1)
    imax(xi[:,:,ns],axe[0,0],cmap='jet',titler=loader.idvar[6])
    imax(yi[:,:,0],axe[1,0],cmap='seismic',titler=loader.file)
    fig.canvas.draw_idle()
sfreq.on_changed(update)


def replot(val):
    global yr,xr
    ys = yr>val
    xs = xr[:,:,:5].sum(axis=2)
    xs = utils.MinMaxScalar(xs)
    imax(ys,axe[1,2],cmap="Greens",titler="Thresholded GT")
    imax(xs>val,axe[0,2],titler="Prediction MT")
    fig.canvas.draw_idle()
#samp.on_changed(replot)


resetax = plt.axes([0.025, 0.1, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sfreq.reset()
    samp.reset()
#	ax.margins(0.1)
button.on_clicked(reset)


dax = plt.axes([0.025, 0.8, 0.15, 0.05], facecolor=axcolor)
text = TextBox(dax,"Species", initial=6)
def selspec(val):
    global ns
    ns = int(val)
    loader.selSpec(ns)
    T = loader.getMat()
    imax(T,axe[0,0],cmap='jet',titler=loader.idvar[ns])

text.on_submit(selspec)


sampler= utils.vectorise

rax1 = plt.axes([0.025, 0.4, 0.15, 0.15], facecolor=axcolor)
radio1 = RadioButtons(rax1, (6,12,24,48), active=2)
def colorfunc(val):
    global xr, yr, sampler
    pool = int(val)
    xr = utils.preporcess(xi,pool=pool,sampler=sampler)
    yr = utils.preporcess(np.abs(yi),
                          pool=pool,
                           scalar=utils.MinMaxScalar)
    imax(yr,axe[1,1],cmap="Greens",titler="Ground Truth")
    xs = xr[:,:,:5].sum(axis=2)
    imax(xs,axe[0,1],titler="Features",cmap="Reds_r")
    fig.canvas.draw_idle()
radio1.on_clicked(colorfunc)


slister = ['mean','maxp','covariance','kurtosis']
rax2 = plt.axes([0.025, 0.6, 0.15, 0.15], facecolor=axcolor)
radio2 = RadioButtons(rax2, slister, active=2)
def chfeats(val):
    global sampler
    if 'm' in val:
        sampler = utils.maverages
    else:
        sampler = utils.vectorise
    
radio2.on_clicked(chfeats)


tata = model.Unsup()

models = ['svm','lof','isf','een'] 
modict = {s:i for i,s in enumerate(models)}
rax3 = plt.axes([0.025, 0.15, 0.15, 0.2], facecolor=axcolor)
radio3 = RadioButtons(rax3, models, active=2,activecolor='red')
def chmodel(val):
    global xr,yr
    i = modict[val]
    tata.train(xr,yr,i=i)
    ys = tata.bscore(xr)
    namer = tata.ana.__class__.__name__
    imax(ys,axe[0,2],titler=namer)
    #axe[1,2].remove()
    cm = test.confusion_matrix(yr>0.5,ys)
    axe[1,2] = sns.heatmap(cm, annot=True,ax=axe[1,2],cmap=plt.cm.Blues,cbar=False)
    fig.canvas.draw_idle()
    
radio3.on_clicked(chmodel)










#rax2 = plt.axes([0.85, 0.3, 0.1, 0.15], facecolor=axcolor)
#radio2 = RadioButtons(rax2, ('covariance', 'kurtosis','none','both'), active=0)



plt.show()


