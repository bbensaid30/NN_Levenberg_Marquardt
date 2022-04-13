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

SMALL_SIZE=8
MEDIUM_SIZE=10
BIGGER_SIZE=12

plt.rc("xtick",labelsize=SMALL_SIZE)
plt.rc("ytick",labelsize=SMALL_SIZE)

def identical(list):
    if(len(list)==0):
        return False
    element=list[0]
    for e in list:
        if(np.abs(e-element)>10**(-3)):
            return False
    return True

def histosRegression(pathInfo):
    iters=[]; times=[]; costsTrain=[]; costsTest=[]; energies=[]

    fileContent=pd.read_csv(pathInfo,header=None).to_numpy()
    draw = (fileContent.shape[0]-2)//8

    for i in range(draw):
        iters.append(fileContent[8*i+1][0])
        times.append(fileContent[8*i+2][0])
        costsTrain.append(fileContent[8*i+3][0])
        costsTest.append(fileContent[8*i+5][0])
        energies.append(fileContent[8*i+7][0])
    
    nonConv = fileContent[8*draw][0]
    div = fileContent[8*draw+1][0]
    
    plt.figure(0)
    plt.title("Distribution of the number of iterations("+str(nonConv*100)+"% not enough iterations, " + str(div*100) + "% divergence)")
    plt.xlabel("iters")
    plt.ylabel("Probability")
    if(identical(iters)):
            plt.axvline(x=iters[0],ymin=0,ymax=1)
    else:
        sns.histplot(iters, stat='probability')
        #sns.kdeplot(iters)
        #plt.axvline(x=0,ymin=0,ymax=1,color="red")

    plt.figure(1)
    plt.title("Distribution of the execution time")
    plt.xlabel("Execution time")
    plt.ylabel("Probability")
    if(identical(times)):
            plt.axvline(x=times[0],ymin=0,ymax=1)
    else:
        sns.histplot(times, stat='probability')

    plt.figure(2)
    plt.title("Distribution of the training cost")
    plt.xlabel("Training cost")
    plt.ylabel("Probability")
    sns.histplot(costsTrain, stat='probability')

    plt.figure(3)
    plt.title("Distribution of the testing cost")
    plt.xlabel("Testing cost")
    plt.ylabel("Probability")
    sns.histplot(costsTest, stat='probability')

    plt.figure(4)
    plt.title("Energetic inequalities")
    plt.xlabel("propE")
    plt.ylabel("Probability")
    if(identical(energies)):
            plt.xlim(-0.05,1.05)
            plt.axvline(x=energies[0],ymin=0,ymax=1)
    else:
        sns.histplot(energies, stat='probability')

    plt.show()



os.chdir("/home/bensaid/Documents/Anabase/NN_shaman") 

folder = "sineWave/"
fileInfo = "info_SGD()15(sigmoid)-1(linear)-(eps=0.001, PTrain=40, PTest=40, tirageMin=0, nbTirages=100, maxIter=200000)Xavier(-10.000000,10.000000,).csv"

pathInfo = "Record/"+folder+fileInfo


histosRegression(pathInfo)


