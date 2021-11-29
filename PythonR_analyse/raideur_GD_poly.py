# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.integrate import odeint

"""
Script sur l'analyse de raideur pour l'équa diff de GD
"""

def g(z):
    # return z**2-1
    # return 2*z**3-3*z**2+5
    return z**4-2*z**2+3

def gp(z):
    # return 2*z
    # return 6*z**2-6*z
    return 4*z**3-4*z

def gpp(z):
    # return 2
    # return 12*z-6
    return 12*z**2-4

def vp_jacobienF(x,y):
    jac = np.zeros((2,2))
    jac[0,0]=gpp(x+y)*g(x+y)+gp(x+y)**2; jac[0,1]=jac[0,0]; jac[1,0]=jac[0,0]
    jac[1,1]=jac[0,0]+gpp(y)*g(y)+gp(y)**2

    eigvals = np.abs(np.linalg.eigvals(jac))
    return np.max(eigvals)/np.min(eigvals)

def raideur(nbTirages,facteur):
    fig = plt.figure(figsize=(10,10))
    #plt.gcf().subplots_adjust(0,0,1,1)
    axes = fig.add_subplot(111)
    axes.set_frame_on(True)
    axes.add_artist(patches.Rectangle((-3,-3),6,6,color="black",fill=False))
    axes.set_xlim(-3,3)
    axes.set_ylim(-3,3)
    axes.set_title(f"Etude de la raideur avec un facteur={facteur}")
    w_raide=[]; b_raide=[]
    w_nraide=[]; b_nraide=[]
    for k in range(nbTirages):
        theta = np.random.uniform(-3,3,2)
        w,b = theta[0], theta[1]
        if(vp_jacobienF(w,b)>facteur):
            w_raide.append(w)
            b_raide.append(b)
        else:
            w_nraide.append(w)
            b_nraide.append(b)
    
    axes.scatter(w_raide,b_raide,color="red",label="Raide")
    axes.scatter(w_nraide,b_nraide,color="blue",label="Non raide")
    axes.legend()

    fig.show()
    plt.show()

raideur(10000,100000)&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
