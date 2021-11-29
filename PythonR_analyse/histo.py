from os import setuid
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import median
import pandas as pd
from scipy import stats
import seaborn as sns
import random as rd

def trace(nbCouples, nbPoints, type):

    x_axis=[]
    for i in range(nbPoints):
        x=int(input("Entrez l'abscisse"))
        x_axis.append(x)
    for k in range(nbCouples):
        y_axis=[]
        eps = float(input("Entrez la valeur de eps"))
        nbTirages = int(input("Entrez le nb de tirages"))
        for i in range(nbPoints):
            y = int(input("Entrez l'ordonnée"))
            y_axis.append(y)
        colors = (rd.random(), rd.random(), rd.random())
        plt.plot(x_axis,y_axis,label="("+str(eps)+","+str(nbTirages)+")", color = colors, marker='o', linestyle='--')

    plt.ylabel("Nombre de mins non équivalents")
    plt.legend()

    if(type=="width"):
        plt.xlabel("largeur")
        plt.title("Evolution du nombre de mins non équivalents en fonction de la largeur")
    elif(type=="deep"):
        plt.xlabel("profondeur")
        plt.title("Evolution du nombre de mins non équivalents en fonction de la profondeur")
    
    plt.show()

def traceVar(nbCouples, type):

    for k in range(nbCouples):
        x_axis=[]
        y_axis=[]
        eps = float(input("Entrez la valeur de eps"))
        nbTirages = int(input("Entrez le nb de tirages"))
        nbPoints = int(input("Nombre de valeurs"))
        for i in range(nbPoints):
            x = int(input("Entrez l'abscisse"))
            x_axis.append(x)
        for i in range(nbPoints):
            y = int(input("Entrez l'ordonnée"))
            y_axis.append(y)
        colors = (rd.random(), rd.random(), rd.random())
        plt.plot(x_axis,y_axis,label="("+str(eps)+","+str(nbTirages)+")", color = colors, marker='o', linestyle='--')

    plt.ylabel("Nombre de mins non équivalents")
    plt.legend()

    if(type=="width"):
        plt.xlabel("largeur")
        plt.title("Evolution du nombre de mins non équivalents en fonction de la largeur")
    elif(type=="deep"):
        plt.xlabel("profondeur")
        plt.title("Evolution du nombre de mins non équivalents en fonction de la profondeur")
    
    plt.show()

def similarCost(cost1,cost1Error,cost2,cost2Error):
    if(cost2-cost2Error>cost1+cost1Error):
        return False
    elif(cost1-cost1Error>cost2+cost2Error):
        return False
    else:
        return True


def newCost(costs,costsError,cost,costError,nbByBins, properValues, properValue):
    taille = len(costs)
    i=0; continuer=True

    if(taille==0):
        costs.append(cost)
        costsError.append(costError)
        nbByBins.append(1)
        properValues.append(properValue)
    else:
        while(i<taille and continuer):
            if(similarCost(cost,costError,costs[i],costsError[i]/nbByBins[i])):
                continuer=False
                costsError[i]+=costError
                nbByBins[i]+=1
            else:
                i+=1
        if(i==taille):
            costs.append(cost)
            costsError.append(costError)
            nbByBins.append(1)
            properValues.append(properValue)
            
    
def histoCost(fileCost, fileProperValues="", type="width", zoom=""):
    CostContent=[]; costs=[]; costsError=[]; nbByBins=[]; bins=[]; properValues=[]
    fileCostContent=pd.read_csv(fileCost,header=None).to_numpy()
    fileProperValuesContent=pd.read_csv(fileProperValues,header=None).to_numpy()
    iter=fileCostContent.shape[0]//3
    print("Nombre de points", iter)

    for i in range(iter):
        if fileCostContent[3*i][0]!=-1:
            newCost(costs,costsError,fileCostContent[3*i][0],fileCostContent[3*i+1][0],nbByBins,properValues,fileProperValuesContent[i][0])
            CostContent.append(fileCostContent[3*i][0])  
    nbHisto=len(costs)
    for i in range(nbHisto):
        costsError[i]/=nbByBins[i]
        nbByBins[i]/=iter

    fig,axes = plt.subplots(1,2,sharex=True,figsize=(10,5))
    try:
        sns.histplot(CostContent, stat='density',ax=axes[0])
    except MemoryError:
        pass
    sns.kdeplot(CostContent, ax=axes[1])
    if(zoom!=""):
        plt.xlim(0,zoom)
    if type=="width":
        fig.suptitle("Répartition des coûts pour neurons="+neuronsString+"("+str(iter)+" points)")
    elif type=="deep":
        fig.suptitle("Répartition des coûts pour L="+LString+"("+str(iter)+" points)")
    else:
        fig.suptitle("Répartition des coûts pour PTrain="+PTrainString+", PTest="+PTestString+"("+str(iter)+" points)")
    axes[0].set_xlabel("Coût"); axes[1].set_xlabel("Coût")

    L = [ (costs[i],i) for i in range(nbHisto) ]
    L.sort()
    sorted_costs,permutation = zip(*L)
    sorted_properValues=[properValues[i] for i in permutation]
    print(sorted_costs)
    
    plt.figure("Instabilité")
    plt.plot(sorted_properValues,sorted_costs,'or')
    if type=="width":
        plt.title("Instabilité en fonction du coût pour neurons="+neuronsString+"("+str(iter)+" points)")
    elif type=="deep":
        plt.title("Instabilité en fonction du coût pour L="+LString+"("+str(iter)+" points)")
    plt.xlabel("Proportion de valeurs propres négatives")
    plt.ylabel("Coût")

    plt.show()
    
    print("Nombre de mins non équivalents environ de: ", nbHisto)
    print("Moyenne des coûts:", np.mean(np.array(CostContent)), " +- ", np.std(np.array(CostContent)))
    print("Min des coûts:", sorted_costs[0])

    return nbHisto

def CVL2(fileCost, fileInputs, fileMoy, function):
    fileBothContent=pd.read_csv(fileCost,header=None).to_numpy()
    iter=fileBothContent.shape[0]//3
    print("Nombre de points", iter)
    costs=[]

    somme=0
    min=100000
    for i in range(iter):
        if fileBothContent[3*i][0]!=-1:
            somme+=fileBothContent[3*i][0]
            if(fileBothContent[3*i][0]<min):
                min=fileBothContent[3*i][0]
            costs.append(fileBothContent[3*i][0])
    """"
    X=pd.read_csv(fileInputs,header=None).to_numpy()
    YPredictMoy=pd.read_csv(fileMoy,header=None).to_numpy()
    YExacte=np.zeros_like(YPredictMoy)
    if function=="sineWave":
        YExacte=0.5+0.25*np.sin(3*np.pi*X)
    elif function=="sinc1":
        YExacte=np.sinc(X/np.pi)
    elif function=="squareWave":
        YExacte=2*(2*np.floor(frequence*X)-np.floor(2*frequence*X))+1
    elif function=="square":
        YExacte=X**2
    elif function=="squareRoot":
        YExacte=np.sqrt(X)

    EMoy=YExacte-YPredictMoy
    """

    print("min:", min)
    print("Moyenne des performances:", somme/iter)
    #print("Performance de la moyenne des prédictions", 0.5*np.sum(np.power(EMoy,2)))
    print("Médiane des coûts:", np.median(np.array(costs)))


algo="LMF"
activationString="reLUl"
#folder="sineWave/P=40|L=2|"+activationString
folder="sineWave/PTrain=100|PTest=100|width=1|"+activationString
fileExtension="1-1-1-1-1-1"
LString="2"
neuronsString="5"
epsString="1e-07"
tirageMinString="0"
nbTiragesString="10000"
#PString="40"
PTrainString="100"
PTestString="100"
fileTraining="Record/"+folder+"/"+"cost_"+algo+"_"+fileExtension+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv"
fileTest="Record/"+folder+"/"+"costTest_"+algo+"_"+fileExtension+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv"
fileBoth="Record/"+folder+"/"+"costBoth_"+algo+"_"+fileExtension+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv"
fileProperValues="Record/"+folder+"/"+"indexeProperValues_"+algo+"_"+fileExtension+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv"
fileInputs="Record/"+folder+"/"+"inputs_"+algo+"_"+fileExtension+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv"
fileMoy="Record/"+folder+"/"+"moy_"+algo+"_"+fileExtension+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv"
#fileName="Record/"+folder+"/"+costString+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv"
#histoCost(fileTest,fileProperValues,"deep")
frequence=1
CVL2(fileBoth,fileInputs,fileMoy,"sineWave")

listFileExtension=[fileExtension]
listFileInputs=[]; listFileBest=[]; listFileMoy=[]
for fE in listFileExtension:
    listFileInputs.append("Record/"+folder+"/"+"inputs_"+algo+"_"+fE+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv")
    listFileBest.append("Record/"+folder+"/"+"best_"+algo+"_"+fE+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv")
    listFileMoy.append("Record/"+folder+"/"+"moy_"+algo+"_"+fE+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv")

def grapheApprox(listFileExtension, listFileInputs,listFileApprox,function):
    nbFiles=len(listFileExtension)
    plt.title("Comparaison entre "+function+" et la sortie du réseau de neurones")
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(nbFiles):
        X=list(pd.read_csv(listFileInputs[i],header=None).to_numpy())
        Y=list(pd.read_csv(listFileBest[i],header=None).to_numpy())
        L = [ (X[i],i) for i in range(len(X)) ]
        L.sort()
        sorted_X,permutation = zip(*L)
        sorted_Y=[Y[i] for i in permutation]
        plt.plot(sorted_X,sorted_Y,label=listFileExtension[i])
    
    YExacte=np.zeros_like(Y)
    sorted_X=np.array(sorted_X)
    if function=="sineWave":
        YExacte=0.5+0.25*np.sin(3*np.pi*sorted_X)
    elif function=="sinc1":
        YExacte=np.sinc(sorted_X/np.pi)
    elif function=="squareWave":
        YExacte=2*(2*np.floor(frequence*sorted_X)-np.floor(2*frequence*sorted_X))+1
    elif function=="square":
        YExacte=sorted_X**2
    elif function=="squareRoot":
        YExacte=np.sqrt(sorted_X)
    plt.plot(sorted_X,YExacte,label=function)
    
    plt.legend()
    plt.show()

#grapheApprox(listFileExtension,listFileInputs,listFileBest,"squareWave")


