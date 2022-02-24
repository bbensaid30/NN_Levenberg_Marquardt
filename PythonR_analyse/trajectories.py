import os
import numpy as np
from numpy.lib.function_base import median
import pandas as pd
from scipy import stats
import seaborn as sns
import random as rd
import os
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.ticker as ticker

def trajectoires(method,filesExtension,colors):
    numColor=0
    for fileExtension in filesExtension:

        fileWeights="Record/weights_"+method+"_"+fileExtension+".csv"
        fileWeightsContent=pd.read_csv(fileWeights,header=None).to_numpy()
        nbIters = fileWeightsContent.shape[0]//2

        iters,w,b=[],[],[]
        for i in range(nbIters):
            iters.append(i)
            w.append(fileWeightsContent[2*i][0])
            b.append(fileWeightsContent[2*i+1][0])
        plt.plot(iters,w,"-",color=colors[numColor],label="w: "+fileExtension)
        plt.plot(iters,b,"--",color=colors[numColor],label="b: "+fileExtension)
        numColor=numColor+1
    
    plt.title("Trajectoires lors des itérations")
    plt.xlabel("itérations")
    plt.legend()

    plt.show()

def profil(method,fileExtension=""):
    fileGradientNorm = "Record/gradientNorm_"+method+"_"+fileExtension+".csv"
    fileCost = "Record/cost_"+method+"_"+fileExtension+".csv"

    fileGradientNormContent=pd.read_csv(fileGradientNorm,header=None).to_numpy()
    fileCostContent=pd.read_csv(fileCost,header=None).to_numpy()
    nbIters = fileCostContent.shape[0]

    iters=[]; gNorms=[]; R=[]
    for i in range(25):
        iters.append(i)
        gNorms.append(fileGradientNormContent[i][0])
        R.append(fileCostContent[i][0])
    plt.axhline(y=R[0],xmin=0,xmax=25,label="R0")

    plt.xlabel("itérations")
    plt.title("Evolution du coût et de la norme du gradient")
    plt.plot(iters,gNorms,label="Norme du gradient", color="r")
    plt.plot(iters,R,label="Coût",color="b")
    plt.legend()
    plt.show()

os.chdir("/home/bensaid/Documents/Anabase/NN_shaman")   

method="Adam"
filesExtension=["(0.3,0.05)","(0.3,0.075)","(0.3,0.11)","(0.3,-0.15)"]
colors=["gold","blue","magenta","orange"]
#trajectoires(method,filesExtension,colors)

fileExtension="(-0.7,1.9)"
profil(method,fileExtension)