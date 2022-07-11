import numpy as np
from numpy.lib.function_base import median
import pandas as pd
from scipy import stats
import scipy as sp
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

def histos(fileGrad,fileDistance,fileIter,fileIterForward,fileInit,minIndices,colorPoints,limits=(-3,3)):
    nbMins = len(minIndices)
    gradContent=[]; grad=[]; gradError=[]; gradBin=[]
    distanceContent=[]; distance=[]; distanceError=[]; distanceBin=[]
    iterContent=[]; iterForwardContent=[]
    for nb in range(nbMins):
        gradContent.append([]); grad.append([]); gradError.append([]); gradBin.append([])
        distanceContent.append([]); distance.append([]); distanceError.append([]); distanceBin.append([])
        iterContent.append([])
        iterForwardContent.append([])

    fileGradContent=pd.read_csv(fileGrad,header=None).to_numpy()
    fileDistanceContent=pd.read_csv(fileGrad,header=None).to_numpy()
    fileIterContent=pd.read_csv(fileIter,header=None).to_numpy()
    fileIterForwardContent=pd.read_csv(fileIterForward,header=None).to_numpy()
    drawSucceed = fileGradContent.shape[0]//3

    for i in range(drawSucceed):
        nbPoint = int(fileGradContent[3*i][0])

        newValue(grad[nbPoint],gradError[nbPoint],fileGradContent[3*i+1][0],fileGradContent[3*i+2][0],gradBin[nbPoint])
        gradContent[nbPoint].append(fileGradContent[3*i+1][0])  
        newValue(distance[nbPoint],distanceError[nbPoint],fileDistanceContent[3*i+1][0],fileDistanceContent[3*i+2][0],distanceBin[nbPoint])
        distanceContent[nbPoint].append(fileDistanceContent[3*i+1][0]) 
        iterContent[nbPoint].append(fileIterContent[2*i+1][0])  
        iterForwardContent[nbPoint].append(fileIterForwardContent[2*i+1][0])  
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
    plt.subplots_adjust(hspace=0.4)
    axes0 = axes0.flatten()
    for nb in range(nbMins):
        sns.histplot(gradContent[nb], stat='density',ax=axes0[nb],log_scale=True)
        if(nb<nbMins-3):
            axes0[nb].set_title("Point: "+minIndices[nb])
        else:
            axes0[nb].set_title(minIndices[nb])
        axes0[nb].tick_params(axis='x',which='both',rotation=45)
    fig0.suptitle("Répartition des normes du gradient pour chacun des minimums "+"("+str(drawSucceed)+" points)")

    #Histogrammes sur la distribution de la distance au vrai minimum lors des tirages
    fig1,axes1 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    plt.subplots_adjust(hspace=0.4)
    axes1 = axes1.flatten()
    for nb in range(nbMins):
        sns.histplot(distanceContent[nb], stat='density',ax=axes1[nb],log_scale=True)
        if(nb<nbMins-3):
            axes1[nb].set_title("Point: "+minIndices[nb])
        else:
            axes1[nb].set_title(minIndices[nb])
        axes1[nb].tick_params(axis='x',which='both',rotation=45)
    fig1.suptitle("Répartition des distances aux minimums pour chacun d'eux "+"("+str(drawSucceed)+" points)")

    #Histogrammes sur la distribution du nombre d'itérations lors des tirages
    fig2,axes2 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    plt.subplots_adjust(hspace=0.4)
    axes2 = axes2.flatten()
    for nb in range(nbMins):
        sns.histplot(iterContent[nb], stat='probability',ax=axes2[nb],log_scale=True)
        if(nb<nbMins-3):
            axes2[nb].set_title("Point: "+minIndices[nb])
        else:
            axes2[nb].set_title(minIndices[nb])
        axes2[nb].tick_params(axis='x',which='both',rotation=45)
    fig2.suptitle("Répartition du nombre d'itérations pour chacun des minimums "+"("+str(drawSucceed)+" points)")

    #Histogrammes sur la distribution du nombre d'itérations forward lors des tirages
    fig3,axes3 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    plt.subplots_adjust(hspace=0.4)
    axes3 = axes3.flatten()
    for nb in range(nbMins):
        sns.histplot(iterForwardContent[nb], stat='probability',ax=axes3[nb],log_scale=True)
        if(nb<nbMins-3):
            axes3[nb].set_title("Point: "+minIndices[nb])
        else:
            axes3[nb].set_title(minIndices[nb])
        axes3[nb].tick_params(axis='x',which='both',rotation=45)
    fig3.suptitle("Répartition du nombre de mises à jour pour chacun des minimums "+"("+str(drawSucceed)+" points)")
    
    fig0.show(); fig1.show(); fig2.show(); fig3.show()
    plt.show()

def init(fileInit,minIndices,colorPoints,example,limits=(-3,3,-3,3),nbIsolines=20, valueR=0.01):
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
    R = np.ma.array(R, mask=np.any([R > valueR], axis=0))

    fig = plt.figure(figsize=(10,10))
    #plt.gcf().subplots_adjust(0,0,1,1)
    axes = fig.add_subplot(111)
    axes.set_frame_on(True)
    axes.add_artist(patches.Rectangle((limits[0],limits[2]),np.abs(limits[1]-limits[0]),np.abs(limits[3]-limits[2]),color="red",fill=False))
    for nb in range(nbMins):
        axes.scatter(initContent[nb][0],initContent[nb][1],c=colorPoints[nb],label=minIndices[nb])
    
    isoLines = axes.contour(W,B,R,nbIsolines)
    #cbar = fig.colorbar(isoLines); cbar.set_label('Isolines', rotation=270)

    axes.legend(bbox_to_anchor=(0.00, 1.0), loc='upper left')
    #axes.legend(bbox_to_anchor=(-0.2, -0.2), loc='lower left')
    #axes.legend(loc='upper left')
    axes.set_xlim(limits[0],limits[1])
    axes.set_xlabel("w")
    axes.set_ylim(limits[2],limits[3])
    axes.set_ylabel("b")
    axes.set_title("Set of initial points that converge to a given minimum")
    
    fig.show()
    plt.show()

def tracking(fileTracking,minIndices):
    nbMins = len(minIndices)
    props=[]; iterContent=[]; propContent=[]; prop_initial_ineq=[]
    for nb in range(nbMins):
        iterContent.append([]); propContent.append([]); prop_initial_ineq.append([])
        props.append(0)
    drawSucceed=0

    fileTrackingContent=pd.read_csv(fileTracking,header=None).to_numpy()
    draw = fileTrackingContent.shape[0]//4

    for i in range(draw):
        nbPoint = int(fileTrackingContent[4*i][0])
        if(nbPoint>=0):
            iterContent[nbPoint].append(fileTrackingContent[4*i+1][0])
            propContent[nbPoint].append(fileTrackingContent[4*i+2][0])
            prop_initial_ineq[nbPoint].append(fileTrackingContent[4*i+3][0])
            props[nbPoint]+=1; drawSucceed+=1
        else:
            iterContent[-nbPoint+len(minIndices)-4].append(fileTrackingContent[4*i+1][0])
            propContent[-nbPoint+len(minIndices)-4].append(fileTrackingContent[4*i+2][0])
            prop_initial_ineq[-nbPoint+len(minIndices)-4].append(fileTrackingContent[4*i+3][0])
            props[-nbPoint+len(minIndices)-4]+=1

    lines = int(np.ceil(nbMins/2))

    fig0,axes0 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    plt.subplots_adjust(hspace=0.4)
    axes0 = axes0.flatten()
    for nb in range(nbMins):
        if(identical(propContent[nb])):
            axes0[nb].set_xlim(-0.05,1.05)
            axes0[nb].axvline(x=propContent[nb][0],ymin=0,ymax=1)
        else:
            sns.histplot(propContent[nb], stat='probability',ax=axes0[nb])
        if(nb<=nbMins-4):
            axes0[nb].set_title("Point: "+minIndices[nb]+" (" + str(round(props[nb]/draw*100,2)) +"%)")
        else:
            axes0[nb].set_title(minIndices[nb]+" (" + str(round(props[nb]/draw*100,2)) +"%)")
        #axes0[nb].set_xlabel("prop_V")
        axes0[nb].set_xlabel("prop_E")
        #axes0[nb].set_xlabel("continuous_entropie")
        axes0[nb].set_ylabel("Probability")
    #fig0.suptitle("Lien entre minimum et diminution de la fonction de Lyapunov "+"("+str(drawSucceed)+" points)")
    fig0.suptitle("Link between minimum and energy")

    fig1,axes1 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    plt.subplots_adjust(hspace=0.4)
    axes1 = axes1.flatten()
    for nb in range(nbMins):
        axes1[nb].scatter(iterContent[nb],propContent[nb])
        if(nb<=nbMins-4):
            axes1[nb].set_title("Point: "+minIndices[nb]+" (" + str(round(props[nb]/draw*100,1)) +"%)")
        else:
            axes1[nb].set_title(minIndices[nb]+" (" + str(round(props[nb]/draw*100,1)) +"%)")
        axes1[nb].set_xlabel("iter")
        axes1[nb].set_ylabel("prop_Em")
    #fig1.suptitle("Lien entre nombre d'itérations et condition entropique "+"("+str(drawSucceed)+" points)")
    fig1.suptitle("Link between energy and number of iterations "+"("+str(drawSucceed)+" convergent points)")

    fig2,axes2 = plt.subplots(lines,2,sharex=True,figsize=(10,10))
    plt.subplots_adjust(hspace=0.4)
    axes2 = axes2.flatten()
    for nb in range(nbMins):
        if(identical(prop_initial_ineq[nb])):
            axes2[nb].set_xlim(-0.05,1.05)
            axes2[nb].axvline(x=prop_initial_ineq[nb][0],ymin=0,ymax=1)
        else:
            sns.histplot(prop_initial_ineq[nb], stat='probability',ax=axes2[nb])
        if(nb<=nbMins-4):
            axes2[nb].set_title("Point: "+minIndices[nb]+" (" + str(round(props[nb]/draw*100,1)) +"%)")
        else:
            axes2[nb].set_title(minIndices[nb]+" (" + str(round(props[nb]/draw*100,1)) +"%)")
        axes2[nb].set_xlabel("initial_inequality")
        axes2[nb].set_ylabel("Probability")
    fig2.suptitle("Link between minimum and connexity and the level sets "+"("+str(drawSucceed)+" convergent points)")

    verif_condition=0
    verif_ineq=0
    for nb in range(nbMins):
        taille = len(propContent[nb])
        for i in range(taille):
            if(np.abs(propContent[nb][i]-1)<10**(-3)):
                verif_condition+=1
            if(np.abs(prop_initial_ineq[nb][i]-1)<10**(-3)):
                verif_ineq+=1
    #print("La proportion de trajectoires vérifiant la condition entropique est: ", verif_condition/drawSucceed)
    print("La proportion de trajectoires vérifiant la diminution d'Em est: ", verif_condition/draw)
    print("La proportion de trajectoires vérifiant la condition 2 est: ", verif_ineq/draw)

    fig0.show(); fig1.show(); fig2.show()
    plt.show()


os.chdir("/home/bensaid/Documents/Anabase/NN_shaman")    


def speedDissipation(fileTab, points, colors):
    plt.title("Etude de la vitesse de dissipation")
    plt.xlabel("Itérations")
    #plt.ylabel("log10(|deltaR/h+grad^2|)")
    #plt.ylabel("log10(|deltaR/h+grad1.grad2|)")
    plt.ylabel("log10(|deltaE/h+v1^2|)")
    nbFiles=len(fileTab)
    for k in range(nbFiles):
        fileName=fileTab[k]
        speedContent = pd.read_csv(fileName,header=None).to_numpy()
        nbModifs = speedContent.shape[0]
        X=[]; Y=[]
        for i in range(nbModifs):
            X.append(i); Y.append(np.log10(np.abs(speedContent[i][0])))
        plt.plot(X,Y, label=points[k], c=colors[k])
    plt.legend()
    plt.show()

def energy_map(fileInit,fileTracking,limits=(-3,3,-3,3)):
    x=[]; y=[]; z=[]

    fileInitContent=pd.read_csv(fileInit,header=None).to_numpy()
    fileTrackingContent=pd.read_csv(fileTracking,header=None).to_numpy()
    draw = fileTrackingContent.shape[0]//4

    for i in range(draw):
        x.append(fileInitContent[3*i+1][0]); y.append(fileInitContent[3*i+2][0])
        z.append(fileTrackingContent[4*i+2][0])
    x=np.array(x);y=np.array(y);z=np.array(z)
    
    fig = plt.figure(figsize=(10,10))
    #plt.gcf().subplots_adjust(0,0,1,1)
    axes = fig.add_subplot(111)
    #axes.set_frame_on(True)
    #axes.add_artist(patches.Rectangle((limits[0],limits[2]),np.abs(limits[1]-limits[0]),np.abs(limits[3]-limits[2]),color="red",fill=False))
    axes.set_xlim(limits[0],limits[1])
    axes.set_ylim(limits[2],limits[3])
    im = axes.scatter(x,y,c=z,cmap=plt.cm.viridis)
    fig.colorbar(im)
    axes.set_xlabel("w")
    axes.set_ylabel("b")

    fig.show()
    plt.show()
    

    """
    xi = np.linspace(x.min(), x.max(), 1000)
    yi = np.linspace(y.min(), y.max(), 1000)

    # Interpolate for plotting
    zi = sp.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='nearest')

    # I control the range of my colorbar by removing data 
    # outside of my range of interest
    zmin = 0
    zmax = 1
    zi[(zi<zmin) | (zi>zmax)] = None

    # Create the contour plot
    CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
                  vmax=zmax, vmin=zmin)
    plt.colorbar()
    plt.xlabel("w")
    plt.ylabel("b")   
    plt.show()
    """

    
example="polyTwo"
limits=(-3,3,-3,3)
nbIsolines=6
valueR=12
minIndices=["(-2,1)","(2,-1)","(0,-1)","(0,1)", "Converge but not to a minimum", "Not enough iterations to converge", "Divergence"]
colorPoints = ["blue","orange","gold","magenta", "gray", "forestgreen", "chocolate"]
#minIndices = ["(2,1)", "(0,-1)", "(-2,3)", "(0,3)", "(-4,3)", "(4,-1)","Eloigné", "Gradient faible", "Divergence"]
#colorPoints=["blue","orange","gold","red","magenta","black", "gray", "forestgreen", "chocolate"]
#minIndices=["(0,-z0)","(0,z0)","Eloigné", "Gradient faible", "Divergence"]
#colorPoints = ["blue","orange", "gray", "forestgreen", "chocolate"]
#minIndices=["(0,z2)","Eloigné", "Gradient faible", "Divergence"]
#colorPoints = ["orange", "gray", "forestgreen", "chocolate"]
algo="EulerRichardson"
setHyperparameters="1"
directory="Record/"+example+"/"+setHyperparameters+"/"+algo+"_"

fileGrad=directory+"gradientNorm.csv"
fileDistance=directory+"distance.csv"
fileIter=directory+"iter.csv"
fileIterForward=directory+"iterForward.csv"
fileInit=directory+"init.csv"
fileTracking=directory+"tracking.csv"
fileTrackingContinuous = directory+"track_continuous.csv"

#histos(fileGrad,fileDistance,fileIter,fileIterForward,fileInit,minIndices,colorPoints)
#init(fileInit,minIndices,colorPoints,example,limits,nbIsolines,valueR)
#tracking(fileTracking,minIndices)

#energy_map(fileInit,fileTracking,limits)

#---------------------------------------------- Pour l'article --------------------------------------------------------------------------------------------

def init_presentation(fileInits,minIndices,colorPoints,example,limits=(-3,3,-3,3),nbIsolines=20, valueR=0.01):
    nbMins = len(minIndices)

    initContents=[]
    for a in range(4):
        initContent=[]
        for nb in range(nbMins):
            initContent.append([[],[]])
        initContents.append(initContent)

    fileInitContents=[]
    for a in range(4):
        fileInitContents.append(pd.read_csv(fileInits[a],header=None).to_numpy())
        draw = fileInitContents[a].shape[0]//3

        for i in range(draw):
            nbPoint = int(fileInitContents[a][3*i][0])
            if(nbPoint>=0):
                initContents[a][nbPoint][0].append(fileInitContents[a][3*i+1][0]);initContents[a][nbPoint][1].append(fileInitContents[a][3*i+2][0])
            else:
                initContents[a][-nbPoint+len(minIndices)-4][0].append(fileInitContents[a][3*i+1][0])
                initContents[a][-nbPoint+len(minIndices)-4][1].append(fileInitContents[a][3*i+2][0])

    W , B = np.meshgrid(np.linspace(limits[0],limits[1],draw//10),np.linspace(limits[2],limits[3],draw//10))
    R = cost(W,B,example)
    R = np.ma.array(R, mask=np.any([R > valueR], axis=0))

    fig = plt.figure(figsize=(10,10))
    plt.gcf().subplots_adjust(wspace=0.2,hspace=0.47)
    axes = fig.subplots(nrows=2,ncols=2)

    for a in range(4):
        if(a==0):
            ax=axes[0,0]
            ax.set_title("(a) GD")
        elif(a==1):
            ax=axes[0,1]
            ax.set_title("(b) Momentum")
        elif(a==2):
            ax=axes[1,0]
            ax.set_title("(c) AWB")
        else:
            ax=axes[1,1]
            ax.set_title("(d) Adam")
        ax.set_frame_on(True)
        ax.add_artist(patches.Rectangle((limits[0],limits[2]),np.abs(limits[1]-limits[0]),np.abs(limits[3]-limits[2]),color="red",fill=False))
        for nb in range(nbMins):
            ax.scatter(initContents[a][nb][0],initContents[a][nb][1],c=colorPoints[nb],label=minIndices[nb])
        isoLines = ax.contour(W,B,R,nbIsolines)
    
        ax.set_xlim(limits[0],limits[1])
        ax.set_xlabel("w")
        ax.set_ylim(limits[2],limits[3])
        ax.set_ylabel("b")
    
    #fig.suptitle("Set of initial points that converge to a given minimum")

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'center', fontsize=6)
    
    fig.show()
    plt.show()

fileInits = ["Record/"+example+"/"+setHyperparameters+"/"+"SGD"+"_init.csv", "Record/"+example+"/"+setHyperparameters+"/"+"Momentum"+"_init.csv", 
"Record/"+example+"/"+setHyperparameters+"/"+"Adam"+"_init.csv", "Record/"+example+"/"+setHyperparameters+"/"+"Adam_bias"+"_init.csv"]
#init_presentation(fileInits,minIndices,colorPoints,example,limits,nbIsolines,valueR)

def init_presentation2(fileInits,minIndices,colorPoints,example,limits=(-3,3,-3,3),nbIsolines=20, valueR=0.01):
    nbMins = len(minIndices)

    initContents=[]
    for a in range(4):
        initContent=[]
        for nb in range(nbMins):
            initContent.append([[],[]])
        initContents.append(initContent)

    fileInitContents=[]
    for a in range(4):
        fileInitContents.append(pd.read_csv(fileInits[a],header=None).to_numpy())
        draw = fileInitContents[a].shape[0]//3

        for i in range(draw):
            nbPoint = int(fileInitContents[a][3*i][0])
            if(nbPoint>=0):
                initContents[a][nbPoint][0].append(fileInitContents[a][3*i+1][0]);initContents[a][nbPoint][1].append(fileInitContents[a][3*i+2][0])
            else:
                initContents[a][-nbPoint+len(minIndices)-4][0].append(fileInitContents[a][3*i+1][0])
                initContents[a][-nbPoint+len(minIndices)-4][1].append(fileInitContents[a][3*i+2][0])


    fig = plt.figure(figsize=(10,10))
    plt.gcf().subplots_adjust(wspace=0.2,hspace=0.47)
    axes = fig.subplots(nrows=2,ncols=2)

    for a in range(4):
        if(a==0):
            ax=axes[0,0]
            ax.set_title("(a) ER/Benchmark 1")
            W , B = np.meshgrid(np.linspace(limits[0],limits[1],draw//10),np.linspace(limits[2],limits[3],draw//10))
            R = cost(W,B,"polyTwo")
            R = np.ma.array(R, mask=np.any([R > 0.1], axis=0))
        elif(a==1):
            ax=axes[0,1]
            ax.set_title("(b) ER/Benchmark 2")
            W , B = np.meshgrid(np.linspace(limits[0],limits[1],draw//10),np.linspace(limits[2],limits[3],draw//10))
            R = cost(W,B,"polyThree")
            R = np.ma.array(R, mask=np.any([R > 12], axis=0))
        elif(a==2):
            ax=axes[1,0]
            ax.set_title("(c) EM/Benchmark 1")
            W , B = np.meshgrid(np.linspace(limits[0],limits[1],draw//10),np.linspace(limits[2],limits[3],draw//10))
            R = cost(W,B,"polyTwo")
            R = np.ma.array(R, mask=np.any([R > 0.1], axis=0))
        elif(a==3):
            ax=axes[1,1]
            ax.set_title("(d) EM/Benchmark 2")
            W , B = np.meshgrid(np.linspace(limits[0],limits[1],draw//10),np.linspace(limits[2],limits[3],draw//10))
            R = cost(W,B,"polyThree")
            R = np.ma.array(R, mask=np.any([R > 12], axis=0))
        ax.set_frame_on(True)
        ax.add_artist(patches.Rectangle((limits[0],limits[2]),np.abs(limits[1]-limits[0]),np.abs(limits[3]-limits[2]),color="red",fill=False))
        for nb in range(nbMins):
            ax.scatter(initContents[a][nb][0],initContents[a][nb][1],c=colorPoints[nb],label=minIndices[nb])
        isoLines = ax.contour(W,B,R,nbIsolines)
    
        ax.set_xlim(limits[0],limits[1])
        ax.set_xlabel("w")
        ax.set_ylim(limits[2],limits[3])
        ax.set_ylabel("b")
    
    #fig.suptitle("Set of initial points that converge to a given minimum")

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="center", fontsize=6)
    
    fig.show()
    plt.show()

fileInits = ["Record/"+"polyTwo"+"/"+setHyperparameters+"/"+"EulerRichardson"+"_init.csv", "Record/"+"polyThree"+"/"+setHyperparameters+"/"
+"EulerRichardson"+"_init.csv","Record/"+"polyTwo"+"/"+setHyperparameters+"/"+"Momentum_Em"+"_init.csv", "Record/"+"polyThree"+"/"+setHyperparameters+"/"
+"Momentum_Em"+"_init.csv"] 
init_presentation2(fileInits,minIndices,colorPoints,example,limits,nbIsolines,valueR)


def energy_map_presentation1(fileInits,fileTrackings,limits=(-3,3,-3,3)):
    fig = plt.figure(figsize=(10,10))
    plt.gcf().subplots_adjust(wspace=0.2,hspace=0.37)
    axes = fig.subplots(nrows=2,ncols=2)

    for a in range(4):
        if(a==0):
            ax=axes[0,0]
            ax.set_title("(a) GD")
        elif(a==1):
            ax=axes[0,1]
            ax.set_title("(b) Momentum")
        elif(a==2):
            ax=axes[1,0]
            ax.set_title("(c) AWB")
        else:
            ax=axes[1,1]
            ax.set_title("(d) Adam")

        x=[]; y=[]; z=[]
        
        fileInitContent=pd.read_csv(fileInits[a],header=None).to_numpy()
        fileTrackingContent=pd.read_csv(fileTrackings[a],header=None).to_numpy()
        draw = fileTrackingContent.shape[0]//4

        for i in range(draw):
            x.append(fileInitContent[3*i+1][0]); y.append(fileInitContent[3*i+2][0])
            z.append(fileTrackingContent[4*i+2][0])
        x=np.array(x);y=np.array(y);z=np.array(z)
    
        ax.set_xlim(limits[0],limits[1])
        ax.set_ylim(limits[2],limits[3])
        ax.set_xlabel("w")
        ax.set_ylabel("b")
        im = ax.scatter(x,y,c=z,cmap=plt.cm.plasma)

    #fig.suptitle("The increasing of the energy along the trajectories")
    cax = fig.add_axes([0.3, 0.50, 0.4, 0.01])
    fig.colorbar(im,cax=cax,orientation='horizontal')

    fig.show()
    plt.show()

fileInits = ["Record/"+example+"/"+setHyperparameters+"/"+"SGD"+"_init.csv", "Record/"+example+"/"+setHyperparameters+"/"+"Momentum"+"_init.csv", 
"Record/"+example+"/"+setHyperparameters+"/"+"Adam"+"_init.csv", "Record/"+example+"/"+setHyperparameters+"/"+"Adam_bias"+"_init.csv"]
fileTrackings = ["Record/"+example+"/"+setHyperparameters+"/"+"SGD"+"_tracking.csv", "Record/"+example+"/"+setHyperparameters+"/"+"Momentum"+"_tracking.csv", 
"Record/"+example+"/"+setHyperparameters+"/"+"Adam"+"_tracking.csv", "Record/"+example+"/"+setHyperparameters+"/"+"Adam_bias"+"_tracking.csv"]

#energy_map_presentation1(fileInits,fileTrackings,limits)

def energy_map_presentation2(fileInits,fileTrackings,limits=(-3,3,-3,3)):
    fig = plt.figure(figsize=(10,10))
    plt.gcf().subplots_adjust(wspace=0.4,hspace=0.37)
    axes = fig.subplots(nrows=1,ncols=2)

    for a in range(2):
        if(a==0):
            ax=axes[0]
            ax.set_title("(a) Benchmark 1")
        elif(a==1):
            ax=axes[1]
            ax.set_title("(b) Benchmark 2")

        x=[]; y=[]; z=[]
        
        fileInitContent=pd.read_csv(fileInits[a],header=None).to_numpy()
        fileTrackingContent=pd.read_csv(fileTrackings[a],header=None).to_numpy()
        draw = fileTrackingContent.shape[0]//4

        for i in range(draw):
            x.append(fileInitContent[3*i+1][0]); y.append(fileInitContent[3*i+2][0])
            z.append(fileTrackingContent[4*i+2][0])
        x=np.array(x);y=np.array(y);z=np.array(z)
    
        ax.set_xlim(limits[0],limits[1])
        ax.set_ylim(limits[2],limits[3])
        ax.set_xlabel("w")
        ax.set_ylabel("b")
        im = ax.scatter(x,y,c=z,cmap=plt.cm.plasma)

    #fig.suptitle("The increasing of the energy along the trajectories")
    cax = fig.add_axes([0.5, 0.11, 0.02, 0.765])
    fig.colorbar(im,cax=cax,orientation='vertical')

    fig.show()
    plt.show()

fileInits = ["Record/"+"polyTwo"+"/"+setHyperparameters+"/"+"EulerRichardson"+"_init.csv", "Record/"+"polyThree"+"/"+setHyperparameters+"/"+"EulerRichardson"+"_init.csv"]
fileTrackings = ["Record/"+"polyTwo"+"/"+setHyperparameters+"/"+"EulerRichardson"+"_tracking.csv", "Record/"+"polyThree"+"/"+setHyperparameters+"/"+"EulerRichardson"+"_tracking.csv"]
#energy_map_presentation2(fileInits,fileTrackings,limits)

nbFiles=10
generalName = "Record/speed_Momentum_Em_"
fileTab=[]
for i in range(1,nbFiles+1):
    fileTab.append(generalName+str(i)+".csv")
points=["(-0.5,0.5)", "(-2.5,2.5)", "(-1.5,0.5)", "(-3,-2.5)", "(-1,-2.5)", "(1,1)", "(-1.1,0)","(1.1,0)", "(2.5,3)","(-5,-4)"]
colors=["blue","orange","gold","magenta", "gray", "forestgreen","chocolate","pink","red","darkviolet"]

#points=["(-0.5,0.5)", "(-2.5,2.5)", "(-1.5,0.5)", "(1,1)", "(-1.1,0)","(1.1,0)","(2,1)"]
#colors=["blue","orange","gold","forestgreen","chocolate","pink","darkviolet"]

#points=["(-0.5,0.5)", "(-2.5,2.5)", "(-1.5,0.5)", "(-3,-2.5)", "(-1,-2.5)", "(1,1)", "(-1.1,0)","(1.1,0)", "(2.5,3)","(2,1)"]
#colors=["blue","orange","gold","magenta", "gray", "forestgreen","chocolate","pink","red","darkviolet"]

#points=["(-0.5,0.5)", "(-2.5,2.5)", "(-1.5,0.5)", "(-1,-2.5)", "(1,1)", "(-1.1,0)","(1.1,0)", "(2.5,3)","(2,1)"]
#colors=["blue","orange","gold", "gray", "forestgreen","chocolate","pink","red","darkviolet"]

#speedDissipation(fileTab,points,colors)



