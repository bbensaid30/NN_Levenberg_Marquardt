from os import setuid
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import median
import pandas as pd

def trace(nbPoints, type):
    x_axis=[]; y_axis=[]
    for i in range(nbPoints):
        x=int(input("Entrez l'abscisse"))
        y=int(input("Entrez l'ordonnée"))
        x_axis.append(x); y_axis.append(y)
    plt.plot(x_axis,y_axis,'or')
    plt.ylabel("Nombre de mins non équivalents")

    if(type=="width"):
        plt.xlabel("width")
        plt.title("Evolution of the number of non equivalent mins as a function of width")
    elif(type=="deep"):
        plt.xlabel("L")
        plt.title("Evolution of the number of non equivalent mins as a function of deep")
    
    plt.show()

def similarCost(cost1,cost1Error,cost2,cost2Error):
    if(cost2-cost2Error>=cost1+cost1Error):
        return False
    elif(cost1-cost1Error>=cost2+cost2Error):
        return False
    else:
        return True


def newCost(costs,costsError,cost,costError,nbByBins):
    taille = len(costs)
    i=0; continuer=True

    if(taille==0):
        costs.append(cost)
        costError.append(costError)
        nbByBins.append(1)
    else:
        while(i<taille and continuer):
            if(similarCost(cost,costError,costs[i],costsError[i])):
                continuer=False
                costError[i]+=costError
                nbByBins[i]+=1
            else:
                i+=1
        if(i==taille):
            costs.append(cost)
            costError.append(costError)
            nbByBins.append(1)
            
    
def histoCost(fileName="", graph=False, seuilProp=0.0):
    costs=[]; costsError=[]; nbByBins=[]; bins=[]
    fileCostContent=pd.read_csv(fileName,header=None).to_numpy()
    iter=fileCostContent.shape[0]%3

    for i in range(iter):
        newCost(costs,costsError,fileCostContent[3*i],fileCostContent[3*i+1],nbByBins)  
    nbHisto=len(costs)
    for i in range(nbHisto):
        costsError[i]/=nbByBins[i]
    
    
    #print(costs)
    amplitudes , bins, patches = plt.hist(CostContent, density=False, range=(0,10), bins=nbHisto, weights=np.ones(iter) / iter)
    #plt.title("Répartition des coûts pour neurons="+neuronsString+"("+str(iter)+" points)")
    plt.title("Répartition des coûts pour L="+LString+"("+str(iter)+" points)")
    #plt.title("Répartition des coûts pour P="+PString+"("+str(iter)+" points)")
    plt.ylabel('Proportion de points')
    plt.xlabel('Coût')
    if(graph):
        plt.show()

    # nbAmplitudes=len(amplitudes)
    nbMinsNonEquivalent=0
    # for i in range(nbAmplitudes):
    #     if amplitudes[i]>seuilProp:
    #         nbMinsNonEquivalent+=1

    for nb in nbByBins:
        if(nb/iter>seuilProp):
            nbMinsNonEquivalent+=1
    
    
    #print("Nombre de mins non équivalents environ de: ", nbMinsNonEquivalent)
    return nbMinsNonEquivalent

def medianHist(fileName="", seuilProp=0.0):
    nbsMins=[]
    ecartHisto=10000
    i=1
    while (ecartHisto!=0):
        largeur=10**(-i)
        nbMin=histoCost(fileName,largeur,seuilProp)
        nbsMins.append(nbMin)
        if (i>=2):
            ecartHisto=nbMin-nbsMins[-2]
        i+=1
        
    
    print("Médiane: ", np.median(np.array(nbsMins)))
    print("Moyenne: ", np.mean(np.array(nbsMins)))
    print("Ecart-type: ", np.std(np.array(nbsMins)))

algo="LMGeodesic"
activationString="sigmoidl"
folder="sineWave/P=40|L"
fileExtension="1-"
LString="2"
neuronsString="1"
epsString="1e-07"
PString="40"
fileName="Record/"+folder+"/cost_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv"
largeur=10**(-3)
seuilProp=0.0
#histoCost(fileName,largeur,True,seuilProp)
medianHist(fileName,seuilProp)

#trace(7,"width")