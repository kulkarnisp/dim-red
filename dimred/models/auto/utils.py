import numpy as np
import os



def rolling_window(data,time_window,pattern_indices=[]):
    ret=[]
    for i in range(len(data)-time_window-1):
        a = data[i:(i+time_window)]
        ret.append(list(a))
    ret=np.array(ret)
    return ret



def normalize_test(xt,yt):
    max_value = np.max(xt)
    min_value = np.min(xt)
    scalar = max_value - min_value
    xt = (xt-min_value)/scalar
    yt = (yt-min_value)/scalar
    return xt,yt



if __name__ =="__main__":
    uset,vset=normalize_test(0,0)
    print("H")