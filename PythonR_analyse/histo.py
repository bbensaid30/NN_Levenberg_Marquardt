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

def histosRegression(pathInfo, pathNon, type):
    iters=[]; times=[]; costsTrain=[]; costsTest=[]; energies=[]; energiesNonConv=[]; energiesDiv=[]; energiesNoise=[]

    fileContent=pd.read_csv(pathInfo,header=None).to_numpy()
    nonContent=pd.read_csv(pathNon,header=None).to_numpy()
    draw = (fileContent.shape[0]-3)//8
    drawNon = nonContent.shape[0]//2

    for i in range(draw):
        iters.append(fileContent[8*i+1][0])
        times.append(fileContent[8*i+2][0])
        costsTrain.append(fileContent[8*i+3][0])
        costsTest.append(fileContent[8*i+5][0])
        energies.append(fileContent[8*i+7][0])
    for i in range(drawNon):
        if nonContent[2*i][0]==-3:
            energiesDiv.append(nonContent[2*i+1][0])
        elif nonContent[2*i][0]==-2:
            energiesNonConv.append(nonContent[2*i+1][0])
        else:
            energiesNoise.append(nonContent[2*i+1][0])
    
    
    nonConv = fileContent[8*draw][0]
    div = fileContent[8*draw+1][0]
    noise = fileContent[8*draw+2][0]
    
    print("nonConv: ",round(nonConv*100,1))
    print("div: ", round(div*100,1))
    print("noise: ", round(noise*100,1))

    plt.figure(0)
    plt.title("Distribution of the number of iterations")
    plt.xlabel("iters")
    plt.ylabel("Probability")
    if(identical(iters)):
            plt.axvline(x=iters[0],ymin=0,ymax=1)
    else:
        sns.histplot(iters,stat="probability")
    print("min iters: ", min(iters))
    print("max iters: ", max(iters))
    print("median iters: ", np.median(iters))
    print("IQR iters: ", stats.iqr(iters, interpolation = 'midpoint'))

    plt.figure(1)
    plt.title("Distribution of the execution time")
    plt.xlabel("Execution time(s)")
    plt.ylabel("Probability")
    if(identical(times)):
            plt.axvline(x=times[0],ymin=0,ymax=1)
    else:
        sns.histplot(times, stat='probability')
    print("min times: ", min(times))
    print("max times: ", max(times))
    print("median times: ", np.median(times))
    print("IQR times: ", stats.iqr(times, interpolation = 'midpoint'))

    plt.figure(2)
    plt.title("Distribution of the training cost")
    plt.xlabel("Training cost")
    if(identical(costsTrain)):
        plt.ylabel("Probability")
        plt.axvline(x=costsTrain[0],ymin=0,ymax=1)
    else:
        if type=="histo":
            plt.ylabel("Probability")
            sns.histplot(costsTrain,stat="probability")
        else:
            plt.ylabel("Density")
            sns.kdeplot(costsTrain)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    print("min training cost: ", min(costsTrain))
    print("max training cost: ", max(costsTrain))
    print("median training cost: ", np.median(costsTrain))
    print("IQR training cost: ", stats.iqr(costsTrain, interpolation = 'midpoint'))

    plt.figure(3)
    plt.title("Distribution of the testing cost")
    plt.xlabel("Testing cost")
    if(identical(costsTest)):
        plt.ylabel("Probability")
        plt.axvline(x=costsTest[0],ymin=0,ymax=1)
    else:
        if type=="histo":
            plt.ylabel("Probability")
            sns.histplot(costsTest,stat="probability")
        else:
            plt.ylabel("Density")
            sns.kdeplot(costsTest)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    print("min testing cost: ", min(costsTest))
    print("max testing cost: ", max(costsTest))
    print("median testing cost: ", np.median(costsTest))
    print("IQR testing cost: ", stats.iqr(costsTest, interpolation = 'midpoint'))

    plt.figure(4)
    plt.title("Energetic inequalities")
    plt.xlabel("propE")
    plt.ylabel("Probability")
    if(identical(energies)):
            plt.xlim(-0.05,1.05)
            plt.axvline(x=energies[0],ymin=0,ymax=1)
    else:
        sns.histplot(energies, stat='probability')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    plt.figure(5)
    plt.title("Energetic inequalities for divergent trajectories")
    plt.xlabel("propE")
    plt.ylabel("Probability")
    if(identical(energiesDiv)):
            plt.xlim(-0.05,1.05)
            plt.axvline(x=energiesDiv[0],ymin=0,ymax=1)
    else:
        sns.histplot(energiesDiv, stat='probability')
    
    plt.figure(6)
    plt.title("Energetic inequalities for non convergent trajectories")
    plt.xlabel("propE")
    plt.ylabel("Probability")
    if(identical(energiesNonConv)):
            plt.xlim(-0.05,1.05)
            plt.axvline(x=energiesNonConv[0],ymin=0,ymax=1)
    else:
        sns.histplot(energiesNonConv, stat='probability')

    plt.figure(7)
    plt.title("Energetic inequalities for numerical noise trajectories")
    plt.xlabel("propE")
    plt.ylabel("Probability")
    if(identical(energiesNoise)):
            plt.xlim(-0.05,1.05)
            plt.axvline(x=energiesNoise[0],ymin=0,ymax=1)
    else:
        sns.histplot(energiesNoise, stat='probability')

    plt.show()


def histosClassification(pathInfo, pathNon, type):
    iters=[]; times=[]; costsTrain=[]; costsTest=[]; energies=[]; energiesNonConv=[]; energiesDiv=[]; energiesNoise=[]
    classTrain=[]; classTest=[]

    fileContent=pd.read_csv(pathInfo,header=None).to_numpy()
    nonContent=pd.read_csv(pathNon,header=None).to_numpy()
    draw = (fileContent.shape[0]-3)//10
    drawNon = nonContent.shape[0]//2

    for i in range(draw):
        iters.append(fileContent[10*i+1][0])
        times.append(fileContent[10*i+2][0])
        costsTrain.append(fileContent[10*i+3][0])
        costsTest.append(fileContent[10*i+5][0])
        energies.append(fileContent[10*i+7][0])
        classTrain.append(fileContent[10*i+8][0])
        classTest.append(fileContent[10*i+9][0])
    for i in range(drawNon):
        if nonContent[2*i][0]==-3:
            energiesDiv.append(nonContent[2*i+1][0])
        elif nonContent[2*i][0]==-2:
            energiesNonConv.append(nonContent[2*i+1][0])
        else:
            energiesNoise.append(nonContent[2*i+1][0])
    
    
    nonConv = fileContent[10*draw][0]
    div = fileContent[10*draw+1][0]
    noise = fileContent[10*draw+2][0]
    
    print("nonConv: ",round(nonConv*100,1))
    print("div: ", round(div*100,1))
    print("noise: ", round(noise*100,1))

    plt.figure(0)
    plt.title("Distribution of the number of iterations")
    plt.xlabel("iters")
    plt.ylabel("Probability")
    if(identical(iters)):
            plt.axvline(x=iters[0],ymin=0,ymax=1)
    else:
        sns.histplot(iters,stat="probability")
    print("min iters: ", min(iters))
    print("max iters: ", max(iters))
    print("median iters: ", np.median(iters))
    print("IQR iters: ", stats.iqr(iters, interpolation = 'midpoint'))

    plt.figure(1)
    plt.title("Distribution of the execution time")
    plt.xlabel("Execution time(s)")
    plt.ylabel("Probability")
    if(identical(times)):
            plt.axvline(x=times[0],ymin=0,ymax=1)
    else:
        sns.histplot(times, stat='probability')
    print("min times: ", min(times))
    print("max times: ", max(times))
    print("median times: ", np.median(times))
    print("IQR times: ", stats.iqr(times, interpolation = 'midpoint'))

    plt.figure(2)
    plt.title("Distribution of the training cost")
    plt.xlabel("Training cost")
    if(identical(costsTrain)):
        plt.ylabel("Probability")
        plt.axvline(x=costsTrain[0],ymin=0,ymax=1)
    else:
        if type=="histo":
            plt.ylabel("Probability")
            sns.histplot(costsTrain,stat="probability")
        else:
            plt.ylabel("Density")
            sns.kdeplot(costsTrain)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    print("min training cost: ", min(costsTrain))
    print("max training cost: ", max(costsTrain))
    print("median training cost: ", np.median(costsTrain))
    print("IQR training cost: ", stats.iqr(costsTrain, interpolation = 'midpoint'))

    plt.figure(3)
    plt.title("Distribution of the testing cost")
    plt.xlabel("Testing cost")
    if(identical(costsTest)):
        plt.ylabel("Probability")
        plt.axvline(x=costsTest[0],ymin=0,ymax=1)
    else:
        if type=="histo":
            plt.ylabel("Probability")
            sns.histplot(costsTest,stat="probability")
        else:
            plt.ylabel("Density")
            sns.kdeplot(costsTest)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    print("min testing cost: ", min(costsTest))
    print("max testing cost: ", max(costsTest))
    print("median testing cost: ", np.median(costsTest))
    print("IQR testing cost: ", stats.iqr(costsTest, interpolation = 'midpoint'))

    plt.figure(4)
    plt.title("Energetic inequalities")
    plt.xlabel("propE")
    plt.ylabel("Probability")
    if(identical(energies)):
            plt.xlim(-0.05,1.05)
            plt.axvline(x=energies[0],ymin=0,ymax=1)
    else:
        sns.histplot(energies, stat='probability')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    plt.figure(5)
    plt.title("Energetic inequalities for divergent trajectories")
    plt.xlabel("propE")
    plt.ylabel("Probability")
    if(identical(energiesDiv)):
            plt.xlim(-0.05,1.05)
            plt.axvline(x=energiesDiv[0],ymin=0,ymax=1)
    else:
        sns.histplot(energiesDiv, stat='probability')
    
    plt.figure(6)
    plt.title("Energetic inequalities for non convergent trajectories")
    plt.xlabel("propE")
    plt.ylabel("Probability")
    if(identical(energiesNonConv)):
            plt.xlim(-0.05,1.05)
            plt.axvline(x=energiesNonConv[0],ymin=0,ymax=1)
    else:
        sns.histplot(energiesNonConv, stat='probability')

    plt.figure(7)
    plt.title("Energetic inequalities for numerical noise trajectories")
    plt.xlabel("propE")
    plt.ylabel("Probability")
    if(identical(energiesNoise)):
            plt.xlim(-0.05,1.05)
            plt.axvline(x=energiesNoise[0],ymin=0,ymax=1)
    else:
        sns.histplot(energiesNoise, stat='probability')
    
    plt.figure(8)
    plt.title("Distribution of the well classified training data")
    plt.xlabel("proportion of well classified training data")
    plt.ylabel("Probability")
    sns.histplot(classTrain, stat='probability')
    print("min training well classified: ", min(classTrain))
    print("max training well classified: ", max(classTrain))
    print("median training well classified: ", np.median(classTrain))
    print("IQR training well classified: ", stats.iqr(classTrain, interpolation = 'midpoint'))

    plt.figure(9)
    plt.title("Distribution of the well classified testing data")
    plt.xlabel("proportion of well classified testing data")
    plt.ylabel("Probability")
    sns.histplot(classTest, stat='probability')
    print("min testing well classified: ", min(classTest))
    print("max testing well classified: ", max(classTest))
    print("median testing well classified: ", np.median(classTest))
    print("IQR testing well classified: ", stats.iqr(classTest, interpolation = 'midpoint'))

    plt.show()


os.chdir("/home/bensaid/Documents/Anabase/NN_shaman") 

folder = "carreFunction2/"
fileInfo = "info_Adam_bias()15(sigmoid)-2(linear)-(eta=0.001, eps=1e-07, PTrain=50, PTest=50, tirageMin=0, nbTirages=10000, maxIter=200000)uniform(-100.000000,100.000000,).csv"
fileNon = "nonConv_Adam_bias()15(sigmoid)-2(linear)-(eta=0.001, eps=1e-07, PTrain=50, PTest=50, tirageMin=0, nbTirages=10000, maxIter=200000)uniform(-100.000000,100.000000,).csv"
#fileInfo = "info_Momentum_Em()5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-1(linear)-(eps=0.01, PTrain=11272, PTest=2818, tirageMin=0, nbTirages=5000, maxIter=500)Xavier(-10.000000,10.000000,).csv"
#fileNon = "nonConv_Momentum_Em()5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-5(reLU)-1(linear)-(eps=0.01, PTrain=11272, PTest=2818, tirageMin=0, nbTirages=5000, maxIter=500)Xavier(-10.000000,10.000000,).csv"


pathInfo = "Record/"+folder+fileInfo
pathNon = "Record/"+folder+fileNon


histosClassification(pathInfo,pathNon,type="histo")