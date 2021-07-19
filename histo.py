from os import setuid
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import median
import pandas as pd
from scipy import stats

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
        costsError.append(costError)
        nbByBins.append(1)
    else:
        while(i<taille and continuer):
            if(similarCost(cost,costError,costs[i],costsError[i])):
                continuer=False
                costsError[i]+=costError
                nbByBins[i]+=1
            else:
                i+=1
        if(i==taille):
            costs.append(cost)
            costsError.append(costError)
            nbByBins.append(1)
            
    
def histoCost(fileName="", graph=False, type="width"):
    CostContent=[]; costs=[]; costsError=[]; nbByBins=[]; bins=[]
    fileCostContent=pd.read_csv(fileName,header=None).to_numpy()
    iter=fileCostContent.shape[0]//3
    print(iter)

    for i in range(iter):
        newCost(costs,costsError,fileCostContent[3*i][0],fileCostContent[3*i+1][0],nbByBins)
        CostContent.append(fileCostContent[3*i][0])  
    nbHisto=len(costs)
    for i in range(nbHisto):
        costsError[i]/=nbByBins[i]
        nbByBins[i]/=iter
    
    L = [ (costs[i],i) for i in range(nbHisto) ]
    L.sort()
    sorted_cost,permutation = zip(*L)
    sorted_costError=[costsError[i] for i in permutation]
    sorted_nbByBins=[nbByBins[i] for i in permutation]
    print(sorted_cost)
    
    plt.stem(costs, nbByBins,
            markerfmt = 'ro', linefmt = 'g--', basefmt = 'm:', use_line_collection = True)
    plt.margins(0.1, 0.1)
    if type=="width":
        plt.title("Répartition des coûts pour neurons="+neuronsString+"("+str(iter)+" points)")
    elif type=="deep":
        plt.title("Répartition des coûts pour L="+LString+"("+str(iter)+" points)")
    else:
        plt.title("Répartition des coûts pour P="+PString+"("+str(iter)+" points)")
    plt.ylabel('Proportion de points')
    plt.xlabel('Coût')
    if(graph):
        plt.show()
    
    print("Nombre de mins non équivalents environ de: ", nbHisto)
    return 


algo="LMF"
activationString="sigmoidl"
folder="sineWaveSd/P=40|width=1|"+activationString
fileExtension="1-"
LString="1"
neuronsString="1"
epsString="1e-15"
PString="40"
fileName="Record/"+folder+"/cost_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv"
seuilProp=0.0
histoCost(fileName,True,"deep")


#trace(7,"width")