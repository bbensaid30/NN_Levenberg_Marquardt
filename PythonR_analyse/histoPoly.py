
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

def g(z,activation):
    if(activation=="polyTwo"):
        return z**2-1
    elif(activation=="polyThree"):
        return 2*z**3-3*z**2+5
    elif(activation=="polyFour"):
        return z**4-2*z**2+3
    elif(activation=="polyFive"):
        return z**5-4*z**4+2*z**3+8*z**2-11*z-12
    elif(activation=="polyEight"):
        return 35*z**8-360*z**7+1540*z**6-3528*z**5+4620*z**4-3360*z**3+1120*z**2+1
    elif(activation == "cloche"):
        return np.exp(-z**2/2)
    elif(activation == "ratTwo"):
        constante = 2/(3+np.sqrt(5))
        return constante*(1+z**2)/(z**2-2*z+2)

def cost(w,b,example):
    if(example=="polyTwo" or example=="polyThree" or example=="polyFour" or example=="polyFive" or example=="polyEight"):
        return 0.25*(g(w+b,example)**2+g(b,example)**2)
    elif(example == "cloche"):
        return -(np.log(g(b,example)) + np.log(1-g(w+b,example)) + np.log(g(2*w+b,example)) )/3
    elif(example == "ratTwo"):
        return -( np.log(g(b,example)) + np.log(g(w+b,example)) )/2

def similarValue(val1,val1Error,val2,val2Error):
    if(val2-val2Error>val1+val1Error):
        return False
    elif(val1-val1Error>val2+val2Error):
        return False
    else:
        return True

def newValue(values,valuesError,value,valueError,nbByBins):
    taille = len(values)
    i=0; continuer=True

    if(taille==0):
        values.append(value)
        valuesError.append(valueError)
        nbByBins.append(1)
    else:
        while(i<taille and continuer):
            if(similarValue(value,valueError,values[i],valuesError[i]/nbByBins[i])):
                continuer=False
                valuesError[i]+=valueError
                nbByBins[i]+=1
            else:
                i+=1
        if(i==taille):
            values.append(value)
            valuesError.append(valueError)
            nbByBins.append(1)

def histos(fileGrad,fileDistance,fileIter,fileInit,minIndices,colorPoints,limits=(-3,3)):
    nbMins = len(minIndices)
    gradContent=[]; grad=[]; gradError=[]; gradBin=[]
    distanceContent=[]; distance=[]; distanceError=[]; distanceBin=[]
    iterContent=[]
    initContent=[]
    for nb in range(nbMins):
        gradContent.append([]); grad.append([]); gradError.append([]); gradBin.append([])
        distanceContent.append([]); distance.append([]); distanceError.append([]); distanceBin.append([])
        iterContent.append([])
        initContent.append([[],[]])

    fileGradContent=pd.read_csv(fileGrad,header=None).to_numpy()
    fileDistanceContent=pd.read_csv(fileDistance,header=None).to_numpy()
    fileIterContent=pd.read_csv(fileIter,header=None).to_numpy()
    fileInitContent=pd.read_csv(fileInit,header=None).to_numpy()
    drawSucceed = fileGradContent.shape[0]//3

    for i in range(drawSucceed):
        nbPoint = int(fileGradContent[3*i][0])

        newValue(grad[nbPoint],gradError[nbPoint],fileGradContent[3*i+1][0],fileGradContent[3*i+2][0],gradBin[nbPoint])
        gradContent[nbPoint].append(fileGradContent[3*i+1][0])  
        newValue(distance[nbPoint],distanceError[nbPoint],fileDistanceContent[3*i+1][0],fileDistanceContent[3*i+2][0],distanceBin[nbPoint])
        distanceContent[nbPoint].append(fileDistanceContent[3*i+1][0]) 
        iterContent[nbPoint].append(fileIterContent[2*i+1][0])  
        initContent[nbPoint][0].append(fileInitContent[3*i+1][0]);initContent[nbPoint][1].append(fileInitContent[3*i+2][0])  
    for nb in range(nbMins):
        nbHisto=len(grad[nb])
        normalisation = len(gradContent[nb])
        for i in range(nbHisto):
            gradError[nb][i]/=gradBin[nb][i]
            gradBin[nb][i]/=normalisation
    for nb in range(nbMins):
        nbHisto=len(distance[nb])
        normalisation = len(distanceContent[nb])
        for i in range(nbHisto):
            distanceError[nb][i]/=distanceBin[nb][i]
            distanceBin[nb][i]/=normalisation
    
    lines = int(np.ceil(nbMins/2))

    #Histogrammes sur la distribution de la norme du gradient lors des tirages
    fig0,axes0 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    axes0 = axes0.flatten()
    for nb in range(nbMins):
        sns.histplot(gradContent[nb], stat='density',ax=axes0[nb],log_scale=True)
        axes0[nb].set_title("Point: "+minIndices[nb])
        axes0[nb].tick_params(axis='x',which='both',rotation=45)
    fig0.suptitle("Répartition des normes du gradient pour chacun des minimums "+"("+str(drawSucceed)+" points)")

    #Histogrammes sur la distribution de la distance au vrai minimum lors des tirages
    fig1,axes1 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    axes1 = axes1.flatten()
    for nb in range(nbMins):
        sns.histplot(distanceContent[nb], stat='density',ax=axes1[nb],log_scale=True)
        axes1[nb].set_title("Point: "+minIndices[nb])
        axes1[nb].tick_params(axis='x',which='both',rotation=45)
    fig1.suptitle("Répartition des distances aux minimums pour chacun d'eux "+"("+str(drawSucceed)+" points)")

    #Histogrammes sur la distribution du nombre d'itérations lors des tirages
    fig2,axes2 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    axes2 = axes2.flatten()
    for nb in range(nbMins):
        sns.histplot(iterContent[nb], stat='density',ax=axes2[nb],log_scale=True)
        axes2[nb].set_title("Point: "+minIndices[nb])
        axes2[nb].tick_params(axis='x',which='both',rotation=45)
    fig2.suptitle("Répartition du nombre d'itérations pour chacun des minimums "+"("+str(drawSucceed)+" points)")

    fig3 = plt.figure(figsize=(10,10))
    #plt.gcf().subplots_adjust(0,0,1,1)
    axes3 = fig3.add_subplot(111)
    axes3.set_frame_on(True)
    axes3.add_artist(patches.Rectangle(limits,np.abs(limits[1]-limits[0]),np.abs(limits[1]-limits[0]),color="red",fill=False))
    for nb in range(nbMins):
        axes3.scatter(initContent[nb][0],initContent[nb][1],c=colorPoints[nb],label=minIndices[nb])
    axes3.legend()
    axes3.set_xlim(limits[0],limits[1])
    axes3.set_ylim(limits[0],limits[1])
    axes3.set_title("Ensemble des points d'initialisation convergeant vers un certain miminum")
    
    fig0.show(); fig1.show(); fig2.show(); fig3.show()
    plt.show()

def init(fileInit,minIndices,colorPoints,example,limits=(-3,3,-3,3),nbIsolines=20):
    nbMins = len(minIndices)
    initContent=[]
    for nb in range(nbMins):
        initContent.append([[],[]])

    fileInitContent=pd.read_csv(fileInit,header=None).to_numpy()
    draw = fileInitContent.shape[0]//3

    for i in range(draw):
        nbPoint = int(fileInitContent[3*i][0])
        if(nbPoint>=0):
            initContent[nbPoint][0].append(fileInitContent[3*i+1][0]);initContent[nbPoint][1].append(fileInitContent[3*i+2][0])
        else:
            initContent[-nbPoint+len(minIndices)-4][0].append(fileInitContent[3*i+1][0])
            initContent[-nbPoint+len(minIndices)-4][1].append(fileInitContent[3*i+2][0])

    W , B = np.meshgrid(np.linspace(limits[0],limits[1],draw//10),np.linspace(limits[2],limits[3],draw//10))
    R = cost(W,B,example)
    R = np.ma.array(R, mask=np.any([R > 3], axis=0))

    fig = plt.figure(figsize=(10,10))
    #plt.gcf().subplots_adjust(0,0,1,1)
    axes = fig.add_subplot(111)
    axes.set_frame_on(True)
    axes.add_artist(patches.Rectangle((limits[0],limits[2]),np.abs(limits[1]-limits[0]),np.abs(limits[3]-limits[2]),color="red",fill=False))
    for nb in range(nbMins):
        axes.scatter(initContent[nb][0],initContent[nb][1],c=colorPoints[nb],label=minIndices[nb])
    
    isoLines = axes.contour(W,B,R,nbIsolines)
    cbar = fig.colorbar(isoLines)

    axes.legend()
    axes.set_xlim(limits[0],limits[1])
    axes.set_xlabel("w")
    axes.set_ylim(limits[2],limits[3])
    axes.set_ylabel("b")
    axes.set_title("Ensemble des points d'initialisation convergeant vers un certain miminum")
    
    fig.show()
    plt.show()

def tracking(fileTracking,minIndices):
    nbMins = len(minIndices)-3
    iterContent=[]; propContent=[]; prop_initial_ineq=[]
    for nb in range(nbMins):
        iterContent.append([]); propContent.append([]); prop_initial_ineq.append([])
    drawSucceed=0

    fileTrackingContent=pd.read_csv(fileTracking,header=None).to_numpy()
    draw = fileTrackingContent.shape[0]//4

    for i in range(draw):
        nbPoint = int(fileTrackingContent[4*i][0])
        if(nbPoint>=0):
            iterContent[nbPoint].append(fileTrackingContent[4*i+1][0])
            propContent[nbPoint].append(fileTrackingContent[4*i+2][0])
            prop_initial_ineq[nbPoint].append(fileTrackingContent[4*i+3][0])
            drawSucceed+=1

    lines = int(np.ceil(nbMins/2))

    fig0,axes0 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    axes0 = axes0.flatten()
    for nb in range(nbMins):
        if(identical(propContent[nb])):
            axes0[nb].axvline(x=propContent[nb][0],ymin=0,ymax=1)
            axes0[nb].set_ylabel("Probability")
        else:
            sns.histplot(propContent[nb], stat='probability',ax=axes0[nb])
        axes0[nb].set_title("Point: "+minIndices[nb])
        axes0[nb].set_xlabel("prop_entropie")
        #axes0[nb].set_xlabel("continuous_entropie")
    fig0.suptitle("Lien entre minimum et condition entropique "+"("+str(drawSucceed)+" points)")

    fig1,axes1 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    axes1 = axes1.flatten()
    for nb in range(nbMins):
        axes1[nb].scatter(propContent[nb],iterContent[nb])
        axes1[nb].set_title("Point: "+minIndices[nb])
    fig1.suptitle("Lien entre nombre d'itérations et condition entropique "+"("+str(drawSucceed)+" points)")

    fig2,axes2 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    axes2 = axes2.flatten()
    for nb in range(nbMins):
        if(identical(prop_initial_ineq[nb])):
            axes2[nb].axvline(x=prop_initial_ineq[nb][0],ymin=0,ymax=1)
            axes2[nb].set_ylabel("Probability")
        else:
            sns.histplot(prop_initial_ineq[nb], stat='probability',ax=axes2[nb])
        axes0[nb].set_title("Point: "+minIndices[nb])
        axes0[nb].set_xlabel("initial_inequality")
    fig2.suptitle("Lien entre minimum et connexité des sous-niveaux "+"("+str(drawSucceed)+" points)")

    verif_condition=0
    verif_ineq=0
    for nb in range(nbMins):
        taille = len(propContent[nb])
        for i in range(taille):
            if(np.abs(propContent[nb][i]-1)<10**(-3)):
                verif_condition+=1
            if(np.abs(prop_initial_ineq[nb][i]-1)<10**(-3)):
                verif_ineq+=1
    print("La proportion de trajectoires vérifiant la condition entropique est: ", verif_condition/drawSucceed)
    print("La proportion de trajectoires vérifiant la condition 2 est: ", verif_ineq/drawSucceed)

    fig0.show(); fig1.show(); fig2.show()
    plt.show()


os.chdir("/home/bensaid/Documents/Anabase/NN_shaman")    

example="polyThree"
limits=(-3,3,-3,3)
nbIsolines=10
minIndices=["(-2,1)","(2,-1)","(0,-1)","(0,1)", "Eloigné", "Gradient faible", "Divergence"]
colorPoints = ["blue","orange","gold","magenta", "gray", "forestgreen", "chocolate"]
#minIndices = ["(2,1)", "(0,-1)", "(-2,3)", "(0,3)", "(-4,3)", "(4,-1)","Eloigné", "Gradient faible", "Divergence"]
#colorPoints=["blue","orange","gold","red","magenta","black", "gray", "forestgreen", "chocolate"]
#minIndices=["(0,-z0)","(0,z0)","Eloigné", "Gradient faible", "Divergence"]
#colorPoints = ["blue","orange", "gray", "forestgreen", "chocolate"]
#minIndices=["(0,z2)","Eloigné", "Gradient faible", "Divergence"]
#colorPoints = ["orange", "gray", "forestgreen", "chocolate"]
algo="Momentum"
setHyperparameters="1-1"
directory="Record/"+example+"/"+setHyperparameters+"/"+algo+"_"

fileGrad=directory+"gradientNorm.csv"
fileDistance=directory+"distance.csv"
fileIter=directory+"iter.csv"
fileInit=directory+"init.csv"
fileTracking=directory+"tracking.csv"
fileTrackingContinuous = directory+"track_continuous.csv"

#histos(fileGrad,fileDistance,fileIter,fileInit,minIndices,colorPoints)
#init(fileInit,minIndices,colorPoints,example,limits,nbIsolines)
tracking(fileTracking,minIndices)

