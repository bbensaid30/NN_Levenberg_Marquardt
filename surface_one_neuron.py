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

def normalization(X):
    P=X.shape[0]
    dim=1
    mean=np.zeros((dim,1)); standardDeviation=np.zeros((dim,1))

    Xnorm=np.zeros_like(X)

    mean[0,0] = np.mean(X)
    standardDeviation[0,0] = np.std(X)
    print(standardDeviation)
    Xnorm[0] = X[0]-mean[0,0]
    Xnorm[0] /= standardDeviation[0,0]
    
    return Xnorm
    

def forward(w,b,X,g):
    return g(w*X+b)

def error(w,b,X,g,Y):
    E=forward(w,b,X,g)-Y
    return np.inner(E,E)
    
def surfaceNetwork(g,data,P,ex1,ex2,nbPoints, normer=False, plotPoints=False, fileNameCost="", fileNameŴeight=""):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    W = np.linspace(ex1,ex2,num=nbPoints)
    B = np.linspace(ex1,ex2,num=nbPoints)
    W, B = np.meshgrid(W, B)
    L=np.zeros_like(W)

    if data=="SineWave":
        X,Y=dataSineWave(P)
    elif data=="Sinc1":
        X,Y=dataSinc1(P)
    
    if normer:
        X=normalization(X)

    for i in range(nbPoints):
        for j in range(nbPoints):
            L[i,j]=error(W[i,j],B[i,j],X,g,Y)


    if(plotPoints):
        fileWeightsContent=pd.read_csv(fileNameŴeight,header=None,delim_whitespace=True).to_numpy()
        fileCostContent=pd.read_csv(fileNameCost,header=None).to_numpy()
        iter=fileCostContent.shape[0]

        weights=[]; bias=[]; cost=[]
        for i in range(iter):
            weights.append(fileWeightsContent[i][0])
            bias.append(fileWeightsContent[i][1])
            cost.append(fileCostContent[i][0])

    surf = ax.plot_surface(W, B, L, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlim(ex1,ex2); ax.set_ylim(ex1,ex2); ax.set_zlim(0,20); ax.zaxis.set_major_locator(LinearLocator(10)); ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_zlabel("cost")
    ax.set_title("Surface de coût pour P="+PString)

    if(plotPoints):
        ax.scatter3D(weights,bias,cost,'red')

    plt.show()

algo="LMGeodesic"
folder="sinc1|1"
fileExtension=""
epsString="1e-07"
PString="100"
fileNameCost="Record/"+folder+"/cost_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv"
fileNameWeight="Record/"+folder+"/weights_vectors_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv"
surfaceNetwork(sigmoid,"Sinc1",100,-10,10,100,False,False,fileNameCost,fileNameWeight)