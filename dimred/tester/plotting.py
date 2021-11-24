import matplotlib.pyplot as plt


def plot_planar(embd_vector):
    xs = embd_vector[:,:2]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(*xs,marker="o",label="Embedding")
    # ax.scatter(a1,b1,zp,marker="o",label=namer,c='r')
    ax.set_xlabel("eta 1")
    ax.set_ylabel("eta_2")
    ax.set_title("Planar space")


def plot_embedding(embd_vector,color_spec=None,titler="Moment space"):
    xs = embd_vector[:,:3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*xs.T,marker="o",label="Embedding",c=color_spec)
    # ax.scatter(a1,b1,zp,marker="o",label=namer,c='r')
    ax.set_xlabel(f"$eta 1$")
    ax.set_ylabel(f"$eta_2$")
    ax.set_title(titler)