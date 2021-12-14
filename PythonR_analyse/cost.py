import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import median
import pandas as pd
from scipy import stats
import seaborn as sns
import random as rd

def informationFile(PTrain,PTest,L,nbNeurons,activations,type_perte,algo,supParameters,generator,tirageMin,nbTirages,eps,maxIter,fileExtension):
    archi=""
    for l in range(L):
        archi+=str(nbNeurons[l]); archi+="("
        archi+=str(activations[l]); archi+=")"
        archi+="-"
    tailleParameters=len(supParameters)
    gen=generator+"("
    if(tailleParameters!=0):
        for param in range(tailleParameters):
            gen+=str(supParameters[param])
            gen+=","
    gen+=")"

    return algo+"("+fileExtension+")"+archi+"(eps="+str(eps)+", PTrain="+str(PTrain)+", PTest="+str(PTest)+", tirageMin="+str(tirageMin)+", nbTirages="+str(nbTirages)+", maxIter="+str(maxIter)+") "+ gen + " .csv"

def histoCosts(fileCostTrain,fileCostTest):
    CostTrainContent=[];CostTestContent=[]
    fileCostTrainContent=pd.read_csv(fileCostTrain,header=None).to_numpy()
    fileCostTestContent=pd.read_csv(fileCostTest,header=None).to_numpy()
    iter=fileCostTrainContent.shape[0]//3
    print("Nombre de points", iter)

    costMin=10000; indice=0
    iterMoy=0

    for i in range(iter):
        CostTrainContent.append(fileCostTrainContent[3*i+2][0])
        iterMoy+=fileCostTrainContent[3*i+1][0]
        if(fileCostTrainContent[3*i+2][0]<costMin):
            indice=i
            costMin = fileCostTrainContent[3*i+2][0]
        CostTestContent.append(fileCostTestContent[3*i+2][0])

    fig0,axes0 = plt.subplots(1,2,sharex=True,figsize=(10,10))
    try:
        sns.histplot(CostTrainContent, stat='density',ax=axes0[0])
    except MemoryError:
        pass
    sns.kdeplot(CostTrainContent, ax=axes0[1])
    fig0.suptitle("Répartition des coûts d'entraînement")
    axes0[0].set_xlabel("Coût"); axes0[1].set_xlabel("Coût")

    fig1,axes1 = plt.subplots(1,2,sharex=True,figsize=(10,10))
    try:
        sns.histplot(CostTestContent, stat='density',ax=axes1[0])
    except MemoryError:
        pass
    sns.kdeplot(CostTestContent, ax=axes1[1])
    fig1.suptitle("Répartition des coûts de test")
    axes1[0].set_xlabel("Coût"); axes1[1].set_xlabel("Coût")

    fig0.show(); fig1.show()
    plt.show()
    
    print("Min des coûts d'entraînement:", costMin, "dont le coût de test associé vaut: ", fileCostTestContent[3*indice+1][0])
    print("Nombre d'itérations moyenne: ", iterMoy/iter)

os.chdir("/home/bensaid/Documents/Anabase/NN_shaman")  

folder="squareWave"

PTrain=100; PTest=100
L=2; nbNeurons=[3,1]; activations=["reLU","linear"]; type_perte="norme2"
algo="EulerRichardson"
supParameters=[-10,10]; generator="uniform"
tirageMin=0; nbTirages=10000
eps=10**(-3); maxIter="20000"
fileExtension=""

fileEnd = informationFile(PTrain,PTest,L,nbNeurons,activations,type_perte,algo,supParameters,generator,tirageMin,nbTirages,eps,maxIter,fileExtension)
fileEnd = "EulerRichardson()3(reLU)-1(linear)-(eps=0.001, PTrain=100, PTest=100, tirageMin=0, nbTirages=10000maxIter=20000)uniform(-10.000000,10.000000,).csv"
fileCostTrain = "Record/"+folder+"/cost_"+fileEnd
fileCostTest = "Record/"+folder+"/costTest_"+fileEnd

histoCosts(fileCostTrain,fileCostTest)