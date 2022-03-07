# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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




def deriv(y,t,eta,dt):
    
    """
    y : liste contenant les 2 fonctions inconnus 
    t : le temps 
    eta, dt : hyperparametres
    """
    w,b = y 

    # Description des 2 equations differentielles 
    wp = (-2*eta/dt)*gp(w+b)*g(w+b) 
    bp = (-2*eta/dt)*(gp(w+b)*g(w+b)+gp(b)*g(b))

    return wp,bp

def eulerExplicite(deriv,debut,fin,y0,nombrePoints,eta,dt):
    pasTemps=(fin-debut)/nombrePoints
    yt = np.asarray(y0)
    t=0
    y_list = np.zeros((nombrePoints+1,2))
    y_list[0,:] = yt
    for k in range(nombrePoints):
        yt+=pasTemps*np.asarray(deriv(yt,t,eta,dt))
        t+=pasTemps
        y_list[k+1,:] = yt

    return y_list
    
y0 = 2.4,-0.3

debut=0
fin=1
nombrePoints=100
time=np.linspace(debut,fin,nombrePoints)

# Paramètres du modèle 
eta = 0.1
dt = 0.01

#Résolution
sol = odeint(deriv, y0, time, args = ( eta, dt))
w,b = sol.T
#sol = eulerExplicite(deriv,debut,fin,y0,nombrePoints,eta,dt)
#w,b = sol[1:,0],sol[1:,1]


plt.figure(figsize=(20,10))
plt.plot(time, w, label="w")
plt.plot(time, b,label="b")

plt.xlabel("temps")
plt.ylabel("Poids")
plt.legend()
plt.title(f"Evolution des poids pour polyTwo et pour algo=GD avec eta = {eta} et dt = {dt}")

plt.show()
