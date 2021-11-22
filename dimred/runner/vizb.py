
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from ipywidgets import interact

import data
import utils
import model
import test


import sys

fname = 'hcci2'
if len(sys.argv)>1:
    fname = sys.argv[1]

time = 3
if len(sys.argv)>2:
    time = sys.argv[2]

if len(sys.argv)>3:
    pool = sys.argv[3]


scorer = 'direct'


dpath = f"/home/shubham/Projects/Anomaly/data/{fname}.npy"
data =np.transpose(np.load(dpath),(1,2,0))

#3 data =np.transpose(np.load(dpath),(0,2,1,3))

tata = model.Unsup()
#cv = utils.co_variance
models = {i:c.__class__.__name__ for i,c in enumerate(tata.clfs)}



##def setfig()
fig,ax = plt.subplots(3,2)#,figsize=(18,6))

axamp = plt.axes([0.25, .03, 0.50, 0.02])
# Slider
samp = Slider(axamp, 'Amp', 5, 40, valinit=10,valstep=5)

def update(val):
    # amp is the current value of the slider
    amp = samp.val
    # update curve
    combine(pool=amp)
    # redraw canvas while idle
    fig.canvas.draw_idle()

# call update function on slider value change
samp.on_changed(update)


def combine(d1 = data,pool=10,time=time):
    global scorer,ax
##    fig,ax = plt.subplots(2,3)#,figsize=(18,6))
    
    ax1 = ax[0,0]
    ax1.imshow(d1[:,:,0],cmap='jet')
#    ax1.set_title(f'Temperature_{0.361+0.001*time}')
    ax1.set_title(f'Temperature_at time {time}')

    ax2 = ax[0,1]
    xs = utils.preporcess(d1,pool=pool,sampler=utils.vectorise)
    ax2.imshow(xs.sum(axis=2),cmap="Reds_r")
    ax2.set_title('Featurised')
    
    loc  = [1,1,2,2]
    for i in range(4):
        tata.train(xs,i=i)
        namer = tata.ana.__class__.__name__
        y = tata.bscore(xs)
#        ax1 = ax[i%2,loc[i]]
        ax1 = ax[loc[i],i%2]
        ax1.imshow(y,cmap="Reds")
        ax1.set_title(namer)
    
    plt.show()





'''


slister = ['direct','threshold']
rax2 = plt.axes([0.025, 0.6, 0.15, 0.15], facecolor='lightgoldenrodyellow')
radio2 = RadioButtons(rax2, slister, active=2)
def chfeats(val):
    global scorer
    scorer = val  


'''

print(__name__)

print(__name__ == "__main__")



# print(__name__)
if __name__ == "__main__":
    a =0
#   genfig()
    combine(time=time)
    ## combine(data[8]-data[7])
