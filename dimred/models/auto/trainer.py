
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable

# from model import AutoEncoder
from utils import rolling_window



class Trainer():
    def __init__(self,netr,x=None,time_window=10, lr=0.01):
        self.tw = time_window
        self.x = self.process(x,time_window)
        self.net = netr # AutoEncoder(time_window)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        
    def process(self,x,time_window):
        xtrain = rolling_window(x,time_window)
        xtrain = xtrain.reshape(-1,1,time_window)
        return torch.from_numpy(xtrain).float()
        
    def train(self,Nmax=500):
        self.net_loss=[]
        for e in range(Nmax):
            var_x = Variable(self.x)
            out = self.net(var_x)
            loss = self.criterion(out, var_x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.net_loss.append(loss.item())
            if (e + 1) % 50 == 0:
                print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

    def eval(self,x=0):
        net = self.net.eval()
        trainx = self.x
        if type(x)!=int:
            trainx = self.process(x,self.tw)
        xans = net(trainx).detach().cpu().numpy()
        return xans[:,0,1]        
    
    def plott(self):
        self.xi = self.x.detach().numpy()[:,0,0]
        self.xo = self.eval()
        plt.plot(xi,label="Input")
        plt.plot(xo,label="Reconstruction")
        plt.title("Train data reconstruction")
        plt.xlabel("")
        plt.show()
        
    def plote(self):
#         n = 
        plt.plot((self.xi-self.xe)**2,label="Error")
        plt.title("Reconstruction Error")
        plt.show()
        
