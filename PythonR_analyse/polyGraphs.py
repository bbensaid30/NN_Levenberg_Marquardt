import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pandas as pd

def polyTwo(fileExtension, c=0):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()
    W = np.linspace(-2,2,num=100)
    B = np.linspace(-2,2,num=100)
    W, B = np.meshgrid(W, B)
    L = (((W+B)**2-1)**2 + (B**2-1)**2)*np.exp(c)
    
    surf = ax.plot_surface(W, B, L, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_zlim(0,10); ax.zaxis.set_major_locator(LinearLocator(10)); ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_zlabel("cost")
    ax.scatter3D([0,0,-2,2],[-1,1,1,-1],[0,0,0,0])

    fileWeights=open("Record/polyTwo/weights_"+fileExtension+".csv",'r')
    fileCost=open("Record/polyTwo/cost_"+fileExtension+".csv",'r')
    fileMu=open("Record/polyTwo/mu_"+fileExtension+".csv",'r')
    fileWeightsContent=pd.read_csv("Record/polyTwo/weights_"+fileExtension+".csv",header=None).to_numpy()
    fileCostContent=pd.read_csv("Record/polyTwo/cost_"+fileExtension+".csv",header=None).to_numpy()
    fileMuContent=pd.read_csv("Record/polyTwo/mu_"+fileExtension+".csv",header=None).to_numpy()
    iter=fileCostContent.shape[0]

    weights=[]; bias=[]; cost=[]; mu=[]; iters=[]
    for i in range(iter):
        weights.append(fileWeightsContent[2*i][0])
        bias.append(fileWeightsContent[2*i+1][0])
        cost.append(fileCostContent[i][0])
        mu.append(fileMuContent[i][0])
        iters.append(i)

    ax.plot3D(weights,bias,cost,'red')
    ax.scatter3D(weights,bias,cost,'red')

    ax2.set_xlabel("iterations")
    ax2.set_ylabel("cost")
    ax2.plot(iters,cost,'o-b')
    ax2.hlines(y=0,xmin=0,xmax=iter,color="red",label="global minimum")

    ax3.set_xlabel("iterations")
    ax3.set_ylabel("mu")
    ax3.plot(iters,mu,'o-b')
    
    plt.show()

def polyThree(fileExtension, c=0):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()
    W = np.linspace(-1,1,num=100)
    B = np.linspace(-1,1,num=100)
    W, B = np.meshgrid(W, B)
    L = ((2*(W+B)**3-3*(W+B)**2+5)**2 + (2*B**3-3*B**2+5)**2)*np.exp(c)
    
    surf = ax.plot_surface(W, B, L, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_zlim(0,30); ax.zaxis.set_major_locator(LinearLocator(10)); ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_zlabel("cost")
    ax.scatter3D([0,2,-2,-1,0],[-1,-1,1,0,1],[0,16*np.exp(c),16*np.exp(c),25*np.exp(c),32*np.exp(c)])

    fileWeights=open("Record/polyThree/weights_"+fileExtension+".csv",'r')
    fileCost=open("Record/polyThree/cost_"+fileExtension+".csv",'r')
    fileMu=open("Record/polyThree/mu_"+fileExtension+".csv",'r')
    fileWeightsContent=pd.read_csv("Record/polyThree/weights_"+fileExtension+".csv",header=None).to_numpy()
    fileCostContent=pd.read_csv("Record/polyThree/cost_"+fileExtension+".csv",header=None).to_numpy()
    fileMuContent=pd.read_csv("Record/polyThree/mu_"+fileExtension+".csv",header=None).to_numpy()
    iter=fileCostContent.shape[0]

    weights=[]; bias=[]; cost=[]; mu=[]; iters=[]
    for i in range(iter):
        weights.append(fileWeightsContent[2*i][0])
        bias.append(fileWeightsContent[2*i+1][0])
        cost.append(fileCostContent[i][0])
        mu.append(fileMuContent[i][0])
        iters.append(i)

    ax.plot3D(weights,bias,cost,'red')
    ax.scatter3D(weights,bias,cost,'red')

    ax2.set_xlabel("iterations")
    ax2.set_ylabel("cost")
    ax2.plot(iters,cost,'o-b')
    ax2.hlines(y=0,xmin=0,xmax=iter,color="red",label="global minimum")
    ax2.hlines(y=16*np.exp(c),xmin=0,xmax=iter,color="green",label="local minimum 1")
    ax2.hlines(y=25*np.exp(c),xmin=0,xmax=iter,color="green",label="local minimum 2")
    ax2.hlines(y=32*np.exp(c),xmin=0,xmax=iter,color="green",label="local minimum 3")
    
    ax3.set_xlabel("iterations")
    ax3.set_ylabel("mu")
    ax3.plot(iters,mu,'o-b')
    
    plt.show()

def polyFour(fileExtension, c=0):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()
    W = np.linspace(-2,2,num=100)
    B = np.linspace(-2,2,num=100)
    W, B = np.meshgrid(W, B)
    L = (((W+B)**4-2*(W+B)**2+3)**2 + (B**4-2*B**2+3)**2)*np.exp(c)
    
    surf = ax.plot_surface(W, B, L, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.zaxis.set_major_locator(LinearLocator(10)); ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_zlabel("cost")
    ax.scatter3D([0,0,-2,2],[-1,1,1,-1],[8*np.exp(c),8*np.exp(c),8*np.exp(c),8*np.exp(c)])

    fileWeights=open("Record/polyFour/weights_"+fileExtension+".csv",'r')
    fileCost=open("Record/polyFour/cost_"+fileExtension+".csv",'r')
    fileMu=open("Record/polyFour/mu_"+fileExtension+".csv",'r')
    fileWeightsContent=pd.read_csv("Record/polyFour/weights_"+fileExtension+".csv",header=None).to_numpy()
    fileCostContent=pd.read_csv("Record/polyFour/cost_"+fileExtension+".csv",header=None).to_numpy()
    fileMuContent=pd.read_csv("Record/polyFour/mu_"+fileExtension+".csv",header=None).to_numpy()
    iter=fileCostContent.shape[0]

    weights=[]; bias=[]; cost=[]; mu=[]; iters=[]
    for i in range(iter):
        weights.append(fileWeightsContent[2*i][0])
        bias.append(fileWeightsContent[2*i+1][0])
        cost.append(fileCostContent[i][0])
        mu.append(fileMuContent[i][0])
        iters.append(i)

    ax.plot3D(weights,bias,cost,'red')
    ax.scatter3D(weights,bias,cost,'red')

    ax2.set_xlabel("iterations")
    ax2.set_ylabel("cost")
    ax2.plot(iters,cost,'o-b')
    ax2.hlines(y=8*np.exp(c),xmin=0,xmax=iter,color="red")

    ax3.set_xlabel("iterations")
    ax3.set_ylabel("mu")
    ax3.plot(iters,mu,'o-b')
    
    plt.show()


polyFour("LMGeodesic_")