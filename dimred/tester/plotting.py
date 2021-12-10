import matplotlib.pyplot as plt
# from 
import numpy as np

# def plot_compare(xold,xnew,):
#     pass


def plot_compare(xold,xnew,titler="Species -X",species=2):
    avg_err = np.mean((xold-xnew)**2)
    xold = xold[:,species]
    xnew = xnew[:,species]
    fig,ax = plt.subplots(ncols=2,figsize=(13,6))
    ax[0].plot(xold,label="Origin")
    ax[0].plot(xnew,label="Reconstr")
    err = np.mean((xold-xnew)**2)
    print(f"Average Error is {avg_err:.4f}")
    ax[0].set_title(titler+f"{species} err:{err:.4f}")
    ax[1].plot(np.abs(xold-xnew),"g")
    ax[1].set_title("Reconstruction Error")

    ax[0].legend()
    plt.show()
    return err


def plot_planar(embd_vector):
    xs = embd_vector[:,:2]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(*xs,marker="o",label="Embedding")
    # ax.scatter(a1,b1,zp,marker="o",label=namer,c='r')
    ax.set_xlabel("eta 1")
    ax.set_ylabel("eta_2")
    ax.set_title("Planar space")


def plot_embedding(embd_vector,color_spec=None,cmap="viridis",titler="Moment space"):
    xs = embd_vector[:,:3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*xs.T,marker="o",label="Embedding",c=color_spec,cmap=cmap)
    # ax.scatter(a1,b1,zp,marker="o",label=namer,c='r')
    ax.set_xlabel(f"$eta 1$")
    ax.set_ylabel(f"$eta_2$")
    ax.set_title(titler)