import os
fpath ="/home/shubham/strial/Combust/hcci/featdat/"
flist = os.listdir(fpath)

ndx = []
for f in flist:
    ndx.append(np.load(fpath+f))
ndx = np.array(ndx)

n,_,_ = ndx.shape
t = ndx[:,:,1].reshape(n,-1)