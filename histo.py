import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def newCost(costs,cost,largeur):
    taille = len(costs)
    i=0; continuer=True

    while (i<taille and continuer):
        if(abs(costs[i]-cost)>largeur):
            i+=1
        else:
            continuer=False
    if(i==taille):
        costs.append(cost)
    
def histoCost(fileName="", largeur=10**(-3)):
    costs=[]; CostContent=[]
    fileCostContent=pd.read_csv("Record/cost_"+fileName,header=None).to_numpy()
    iter=fileCostContent.shape[0]

    for i in range(iter):
        newCost(costs,fileCostContent[i][0],largeur)
        CostContent.append(fileCostContent[i][0])
    
    nbHisto=len(costs)
    print("Le nombre de mins non équivalents est d'environ ", nbHisto)
    plt.hist(CostContent, density=True, bins=nbHisto) 
    plt.title("Proportion des coûts")
    plt.ylabel('Fréquence')
    plt.xlabel('Coût')

    plt.show()

fileName="LMGeodesic_sineWave(eps=1e-7).csv"
largeur=10**(-3)
histoCost(fileName,largeur)