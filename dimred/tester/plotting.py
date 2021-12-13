import matplotlib.pyplot as plt
# from 
import numpy as np
import pandas as pd

# def plot_compare(xold,xnew,):
#     pass


def plot_compare(xold,xnew,titler="Species -X",species=2,labels=["Origin","Reconstruct"]):
    avg_err = np.mean((xold-xnew)**2)
    xold = xold[:,species]
    xnew = xnew[:,species]
    fig,ax = plt.subplots(ncols=2,figsize=(13,6))
    ax[0].plot(xold,label=labels[0])
    ax[0].plot(xnew,label=labels[1])
    err = np.mean((xold-xnew)**2)
    print(f"Average Error is {avg_err:.4f}")
    ax[0].set_title(titler+f"{species} err:{err:.4e}")
    ax[1].plot(np.abs(xold-xnew),"g")
    ax[1].set_title(f"Error Comparison")

    ax[0].legend()
    return fig


def plot_planar(embd_vector,color_spec=None,cmap="viridis",titler="Planar space",ax=None):
    xs = embd_vector[:,:2]
    if ax ==None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.scatter(*xs,marker="o",label="Embedding",cmap=cmap)
    # ax.scatter(a1,b1,zp,marker="o",label=namer,c='r')
    ax.set_xlabel("eta 1")
    ax.set_ylabel("eta_2")
    ax.set_title(f"{titler}")


def plot_embedding(embd_vector,color_spec=None,cmap="viridis",titler="Moment space",ax=None):
    xs = embd_vector[:,:3]
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*xs.T,marker="o",label="Embedding",c=color_spec,cmap=cmap)
    # ax.scatter(a1,b1,zp,marker="o",label=namer,c='r')
    ax.set_xlabel(f"$eta 1$")
    ax.set_ylabel(f"$eta_2$")
    ax.set_title(titler)

def plot_spectra(s1,s2,u1,u2):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
    ax1.plot(np.log(s1),'--ob',label='covariance')
    ax1.plot(np.log(s2),'--or',label='cokurtosis')
    ax1.set_ylabel('singular value')
    ax1.legend()
    dots = [abs(v1@v2) for v1,v2 in zip(u1,u2)]
    ax2.plot(dots,'--og')
    ax2.set_ylabel('dot product')
    ax2.set_ylim(0,1.5)
    plt.show()

    
def img_compare(xold,xnew,titler="Species -X",species=2,labels=["Origin","Reconstruct"],aspect=0):
    avg_err = np.mean((xold-xnew)**2)
    print(f"Average Error is {avg_err:.4e}")
    xold = xold[:,:,species]
    xnew = xnew[:,:,species]
    xerr = xold-xnew

    xlist = [xold,xnew,xerr]
    err = np.mean(xerr**2)
    print(f"Species Error is {err:.4e}")

    labels.append(titler+f"{species} err:{err:.4e}")
    fig = plt.figure()

#     fig,ax = plt.subplots(nrows=3,figsize=(13,6))
    cmaps = ['jet','jet','RdBu_r']
    for i,x in enumerate(xlist):
        if aspect==0:
            nx,ny = x.shape
            aspect=ny/nx*0.82
#         img=ax[i].imshow(x,cmap=cmaps[i])
        plt.imshow(x,cmap=cmaps[i])
#         ax[i].set_title(labels[i])
        plt.title(labels[i])
        plt.colorbar()
#         fig.colorbar(img)
        plt.show()
    return fig

def plot_bars(errcv,errck,horz=True,labels=["Covariance","Kurtosis"],indices=None):

    df1 = pd.DataFrame([errcv,errck]).T
    df1.columns = labels
    if indices!= None:
        n = len(errcv)
        temp = list(range(n))
        m = min(len(indices),n)
        temp[:m] = list(indices)[:m]
        df1.index = temp
    if horz:
        df1.plot.barh()#(kind='bar')
    else:
        df1.plot.bar()
    plt.title("Species reconstruction error")
    plt.xlabel("Reconstruction errors in source term $f(x_1,x_2)$")
    plt.ylabel("Reconstruction method")
        