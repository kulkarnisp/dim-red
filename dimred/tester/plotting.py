import matplotlib.pyplot as plt
# from 
import numpy as np

# def plot_compare(xold,xnew,):
#     pass


def plot_compare(xold,xnew,titler="Species -X",iserr=False,species=2):
    xold = xold[:,species]
    xnew = xnew[:,species]
    plt.plot(xold, label="Origin")
    plt.plot(xnew,label="Reconstr")
    err = np.mean((xold-xnew)**2)
    print(f"Error is {err:.4f}")
    plt.title(titler+f"err:{err:.4f}")
    plt.legend()
    plt.show()

    if iserr:
        plt.plot(np.abs(xold-xnew),"g")
        plt.title("Reconstruction Error")
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