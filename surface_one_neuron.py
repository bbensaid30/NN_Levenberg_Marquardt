import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pandas as pd

def sigmoid(inputs):
    return 1/(1 + np.exp(-inputs))

def reLU(inputs):
    return np.maximum(0,inputs)

def dataSineWave(P):
    X=np.linspace(-1,1,num=P)
    Y=0.5+0.25*np.sin(3*np.pi*X)
    return X,Y

def dataSinc1(P):
    X=np.linspace(-1,1,num=P)
    Y=np.sinc(X/np.pi)
    return X,Y

def forward(w,b,X,g):
    return g(w*X+b)

def error(w,b,X,g,Y):
    E=forward(w,b,X,g)-Y
    return np.inner(E,E)
    
def surfaceNetwork(g,data,P,ex1,ex2,nbPoints):
    W = np.linspace(ex1,ex2,num=nbPoints)
    B = np.linspace(ex1,ex2,num=nbPoints)
    W, B = np.meshgrid(W, B)
    L=np.zeros_like(W)

    if data=="SineWave":
        X,Y=dataSineWave(P)
    elif data=="Sinc1":
        X,Y=dataSinc1(P)

    for i in range(nbPoints):
        for j in range(nbPoints):
            L[i,j]=error(W[i,j],B[i,j],X,g,Y)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(W, B, L, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.zaxis.set_major_locator(LinearLocator(10)); ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_zlabel("cost")
    plt.show()

surfaceNetwork(sigmoid,"SineWave",100,-20,20,100)