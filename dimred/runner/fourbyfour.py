import numpy as np
import src

import data,model

loader = data.LoadMPI()
tata = model.Unsup()


def imax(X,ax,cmap="Reds",aspect=1.0,titler=""):
    im = ax.imshow(X,cmap=cmap,aspect=aspect)
#     plt.colorbar(im)
    plt.colorbar(im,ax=ax)
    ax.grid(None)
    ax.set_title(titler)
    ax.set_xticks([])
    ax.set_yticks([])
#     plt.close()
    return ax


def direct(j,pool=24,sampling=utils.vectorise):
    xi,yi = loader.getTrain(j)
    nx,ny,nv = xi.shape

    scaling = utils.MeanMaxScalar
    sampling = utils.maxPool  ## choose form maxPool, Average, coVariance,coKurtosis
    # learning = classifiers list in models module
    xr = utils.preporcess(xi,pool=pool,sampler=sampling)
    yr = utils.preporcess(np.abs(yi),pool=pool,
                          scalar=utils.MinMaxScalar)
    ytrue = yr > 0.5
    fig,axe = plt.subplots(4,5,figsize=(24,16))
    imax(xi[:,:,3],axe[0,0], cmap="jet",titler="Input Data")
    imax(xr.sum(axis=2),axe[1,0], cmap="jet",titler="Features")
   
    imax(yi,axe[2,0], cmap="seismic",titler="Heat Release Rate")
    imax(yr,axe[3,0], cmap="Greens",titler="Ground Truth")
    
    ii = 3
    aucs = []
    times = []
    for jj in range(4):
        ii = jj
        ts =time.time()
        if jj == 3:
            ii+= 1
        tata.train(xr,yr,i=ii)
        # tata.ana = LocalOutlierFactor(novelty=True)
        # yhat = tata.dscore(xr)
        # yhat = tata.bscore(xr)
        # yprob = tata.rscore(xr)
        yprob = tata.rscore(xr)
#         yhat = yprob>(0.5)
        yhat = tata.bscore(xr)
        namer = tata.ana.__class__.__name__
        
        
        aucs.append(roc_auc_score(ytrue.reshape(-1),
                                  yprob.reshape(-1)))
        imax(yhat,ax=axe[jj,2],cmap="Reds",titler=namer)
        imax(yprob,ax=axe[jj,1],cmap="Reds",titler=namer)

        cm = tester.confusion_matrix(ytrue,yhat)
        sns.heatmap(cm,cmap=plt.cm.Blues,ax=axe[jj,3],annot=True)
        te = time.time()
        times.append(te-ts)
        tester.plotROC(yr,yprob,namer=f"ROC{namer}",saver=None,ax=axe[jj,4])
    
    print(namer, ": DOne")
    return [aucs,times]
