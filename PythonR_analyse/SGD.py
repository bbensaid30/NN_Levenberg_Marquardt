# -*- coding: utf-8 -*-

from unicodedata import name
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random as random
import copy as cp

"""
Script sur la résolution du sytème pour GD pour l'exemple polyTwo
"""

def g(z,nameActivation):
    if nameActivation == "polyTwo":
        return z**2-1
    elif nameActivation == "polyThree":
        return 2*z**3-3*z**2+5
    elif nameActivation == "polyFour":
        return z**4-2*z**2+3
    elif nameActivation == "polyFive":
        return z**5-4*z**4+2*z**3+8*z**2-11*z-12
    elif nameActivation == "polyEight":
        return 35*z**8-360*z**7+1540*z**6-3528*z**5+4620*z**4-3360*z**3+1120*z**2+1

def gp(z,nameActivation):
    if nameActivation == "polyTwo":
        return 2*z
    elif nameActivation == "polyThree":
        return 6*z**2-6*z
    elif nameActivation == "polyFour":
        return 4*z**3-4*z
    elif nameActivation == "polyFive":
        return 5*z**4-16*z**3+6*z**2+16*z-11
    elif nameActivation == "polyEight":
        return 280*z**7-2520*z**6+9240*z**5-17640*z**4+18480*z**3-10080*z**2+2240*z

def gpp(z,nameActivation):
    if nameActivation == "polyTwo":
        return 2
    elif nameActivation == "polyThree":
        return 12*z-6
    elif nameActivation == "polyFour":
        return 12*z**2-4
    elif nameActivation == "polyFive":
        return 20*z**3-48*z**2+12*z-16
    elif nameActivation == "polyEight":
        return 1960*z**6-15120*z**5+46200*z**4-70560*z**3+55440*z**2-20160*z+2240

def grad(theta,nameActivation):
    w,b = theta[0,0],theta[1,0]
    gradient = np.zeros((2,1))
    gradient[0,0] = gp(w+b,nameActivation)*g(w+b,nameActivation)
    gradient[1,0] = gradient[0,0]+gp(b,nameActivation)*g(b,nameActivation)
    return 0.5*gradient

def Hessian(theta,nameActivation):
    w,b = theta[0,0],theta[1,0]
    H = np.zeros((2,2))
    H[0,0] = gp(w+b,nameActivation)**2+gpp(w+b,nameActivation)*g(w+b,nameActivation)
    H[0,1] = H[0,0]; H[1,0] = H[0,0]
    H[1,1] = H[0,0]+gp(b,nameActivation)**2+gpp(b,nameActivation)*g(b,nameActivation)
    return 0.5*H

def quasiHessian(theta,nameActivation):
    w,b = theta[0,0],theta[1,0]
    Q = np.zeros((2,2))
    Q[0,0] = gp(w+b,nameActivation)**2
    Q[0,1] = Q[0,0]; Q[1,0] = Q[0,0]
    Q[1,1] = Q[0,0]+gp(b,nameActivation)**2
    return 0.5*Q

def sigma(theta,nameActivation):
    w,b = theta[0,0],theta[1,0]
    sigma = np.zeros((2,2))
    sigma[0,0] = gp(w+b,nameActivation)*g(w+b,nameActivation)
    sigma[1,0] = sigma[0,0]-gp(b,nameActivation)*g(b,nameActivation)
    return 0.5*np.sign(gp(w+b,nameActivation)*g(w+b,nameActivation))*sigma

def eulerExplicite(debut,fin,y0,nombrePoints,eta_bar,b,nameActivation,nbSeed):
    random.seed(nbSeed)
    pasTemps=(fin-debut)/nombrePoints
    yt = cp.deepcopy(y0)
    t=0
    t_list = [t]
    y_list=np.zeros((nombrePoints+1,2))
    y_list[0,:]=yt.flatten()
    for k in range(nombrePoints):
        yt+=-eta_bar*pasTemps*grad(yt,nameActivation)-((eta_bar*np.sqrt(pasTemps))/np.sqrt(b))*np.dot(sigma(yt,nameActivation),np.random.normal(size=(2,1)))
        t+=pasTemps
        t_list.append(t)
        y_list[k+1,:]=yt.flatten()
    return t_list,y_list

def inequality(theta,nameActivation,b,eta_bar):
    hessian = Hessian(theta,nameActivation)
    sig = sigma(theta,nameActivation)
    gradient = grad(theta,nameActivation)
    gauche = np.trace(np.transpose(sig)*hessian*sig)
    droite = np.linalg.norm(gradient)**2
    print("Terme gauche: ", gauche)
    print("Terme droite: ", droite)
    if(gauche<droite):
        print("Vérifié sans la constante")
    else:
        print("Constante nécessaire")
    if(gauche<(2*b*droite)/eta_bar):
        return True
    else:
        return False

def trajectoriesW(debut,fin,y0,nombrePoints,eta_bar,b,nameActivation,nbTirages):
    plt.title("Différentes trajectoires au voisinage d'un minimum")
    plt.xlabel("t")
    plt.ylabel("w")
    for nb in range(nbTirages):
        t_list,y_list = eulerExplicite(debut,fin,y0,nombrePoints,eta_bar,b,nameActivation,nb)
        plt.plot(t_list,y_list[:,0])
    plt.legend()
    plt.show()


nameActivation = "polyTwo"
voisinage=10**(-1)
y0=np.zeros((2,1)); y0[0,0]=0+voisinage; y0[1,0]=1+voisinage

debut=0
fin=2
nombrePoints=20000
time=np.linspace(debut,fin,nombrePoints)

# Paramètres du modèle 
eta_bar=1
b=1

print(inequality(y0,nameActivation,b,eta_bar))
nbTirages=10
trajectoriesW(debut,fin,y0,nombrePoints,eta_bar,b,nameActivation,nbTirages)

