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

def twoSpiral(nbPoints):
    X = np.zeros((2,2*nbPoints))
    Y = np.zeros((1,2*nbPoints))

    theta = np.linspace(0,2*np.pi,num=nbPoints)
    r = 2*theta + np.pi

    for i in range(nbPoints):
        X[0,i] = np.cos(theta[i])*r[i]
        X[1,i] = np.sin(theta[i])*r[i]
        Y[0,i] = 0
    
    for i in range(nbPoints,2*nbPoints):
        X[0,i] = -np.cos(theta[i-nbPoints])*r[i-nbPoints]
        X[1,i] = -np.sin(theta[i-nbPoints])*r[i-nbPoints]
        Y[0,i] = 1
    
    colours=[]
    for i in range(2*nbPoints):
        if Y[0,i]==0:
            colours.append("orange")
        else:
            colours.append("blue")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X[0,:],X[1,:],c=Y[0,:])
    plt.show()

twoSpiral(100)

    