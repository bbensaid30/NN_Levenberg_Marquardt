import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pandas as pd

def polyTwo(fileExtension, c=0):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    W = np.linspace(-1,1,num=100)
    B = np.linspace(-1,1,num=100)
    W, B = np.meshgrid(W, B)
    L = (((W+B)**2-1)**2 + (B**2-1)**2)*np.exp(c)
    
    surf = ax.plot_surface(W, B, L, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_zlim(0,1); ax.zaxis.set_major_locator(LinearLocator(10)); ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_zlabel("cost")
    ax.scatter3D([0,0,-2,2],[-1,1,1,-1],[0,0,0,0])

    fileWeights=open("Record/weights_"+fileExtension+".csv",'r')
    fileCost=open("Record/cost_"+fileExtension+".csv",'r')
    fileWeightsContent=pd.read_csv("Record/weights_"+fileExtension+".csv",header=None).to_numpy()
    fileCostContent=pd.read_csv("Record/cost_"+fileExtension+".csv",header=None).to_numpy()
    iter=fileCostContent.shape[0]

    weights=[]; bias=[]; cost=[]
    for i in range(iter):
        weights.append(fileWeightsContent[2*i][0])
        bias.append(fileWeightsContent[2*i+1][0])
        cost.append(fileCostContent[i][0])

    ax.plot3D(weights,bias,cost,'red')
    
    plt.show()

def polyThree(fileExtension, c=0):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    W = np.linspace(-2,1,num=100)
    B = np.linspace(-1,1,num=100)
    W, B = np.meshgrid(W, B)
    L = ((2*(W+B)**3-3*(W+B)**2+5)**2 + (2*B**3-3*B**2+5)**2)*np.exp(c)
    
    surf = ax.plot_surface(W, B, L, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_zlim(0,300); ax.zaxis.set_major_locator(LinearLocator(10)); ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_zlabel("cost")
    ax.scatter3D([0,2,-2,-1,0],[-1,-1,1,0,1],[0,16*np.exp(c),16*np.exp(c),25*np.exp(c),32*np.exp(c)])

    fileWeights=open("Record/weights_"+fileExtension+".csv",'r')
    fileCost=open("Record/cost_"+fileExtension+".csv",'r')
    fileWeightsContent=pd.read_csv("Record/weights_"+fileExtension+".csv",header=None).to_numpy()
    fileCostContent=pd.read_csv("Record/cost_"+fileExtension+".csv",header=None).to_numpy()
    iter=fileCostContent.shape[0]

    weights=[]; bias=[]; cost=[]
    for i in range(iter):
        weights.append(fileWeightsContent[2*i][0])
        bias.append(fileWeightsContent[2*i+1][0])
        cost.append(fileCostContent[i][0])

    ax.plot3D(weights,bias,cost,'red')
    ax.scatter3D(weights,bias,cost,'red')
    
    plt.show()

#polyTwo("polyTwo")
polyThree("polyThree1")