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

def R(theta,nameActivation):
    w,b = theta[0,0],theta[1,0]
    return 0.25*(g(w+b,nameActivation)**2+g(b,nameActivation)**2)

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

def G(theta,gradient,k,seuilK=1):
    if(k<seuilK):
        r=k
    else:
        r=1/k**2
    result = ((np.sqrt(2)+r)/np.linalg.norm(theta))*np.linalg.norm(gradient)
    return result*np.eye(2)


def expGrad(debut,fin,y0,nombrePoints,std,nameActivation,nbSeed,approxHessian=False):
    random.seed(nbSeed)
    pasTemps=(fin-debut)/nombrePoints
    yt = cp.deepcopy(y0)
    t=0
    t_list = [t]
    y_list=np.zeros((nombrePoints+1,2))
    y_list[0,:]=yt.flatten()
    for k in range(nombrePoints):
        if(approxHessian):
            M = quasiHessian(yt,nameActivation)
        else:
            M = Hessian(yt,nameActivation)
        yt+=-pasTemps/np.linalg.norm(grad(yt,nameActivation))*np.dot(M,grad(yt,nameActivation))+std*np.sqrt(pasTemps)*np.random.normal(size=(2,1))
        t+=pasTemps
        t_list.append(t)
        y_list[k+1,:]=yt.flatten()
    return t_list,y_list

def normalGrad(debut,fin,y0,nombrePoints,std,nameActivation,nbSeed,approxHessian=False):
    random.seed(nbSeed)
    pasTemps=(fin-debut)/nombrePoints
    yt = cp.deepcopy(y0)
    t=0
    t_list = [t]
    y_list=np.zeros((nombrePoints+1,2))
    y_list[0,:]=yt.flatten()
    for k in range(nombrePoints):
        if(approxHessian):
            M = quasiHessian(yt,nameActivation)
        else:
            M = Hessian(yt,nameActivation)
        yt+=-2*pasTemps*np.dot(M,grad(yt,nameActivation))+2*std*np.sqrt(pasTemps)*np.random.normal(size=(2,1))
        t+=pasTemps
        t_list.append(t)
        y_list[k+1,:]=yt.flatten()
    return t_list,y_list

def expR(debut,fin,y0,nombrePoints,std,nameActivation,nbSeed):
    random.seed(nbSeed)
    pasTemps=(fin-debut)/nombrePoints
    yt = cp.deepcopy(y0)
    t=0
    t_list = [t]
    y_list=np.zeros((nombrePoints+1,2))
    y_list[0,:]=yt.flatten()
    for k in range(nombrePoints):
        yt+=-pasTemps*grad(yt,nameActivation)+std*np.sqrt(pasTemps)*np.random.normal(size=(2,1))
        t+=pasTemps
        t_list.append(t)
        y_list[k+1,:]=yt.flatten()
    return t_list,y_list

def normalR(debut,fin,y0,nombrePoints,std,nameActivation,nbSeed):
    random.seed(nbSeed)
    pasTemps=(fin-debut)/nombrePoints
    yt = cp.deepcopy(y0)
    t=0
    t_list = [t]
    y_list=np.zeros((nombrePoints+1,2))
    y_list[0,:]=yt.flatten()
    for k in range(nombrePoints):
        yt+=-2*pasTemps*R(yt,nameActivation)*grad(yt,nameActivation)+2*std*np.sqrt(pasTemps)*np.random.normal(size=(2,1))
        t+=pasTemps
        t_list.append(t)
        y_list[k+1,:]=yt.flatten()
    return t_list,y_list

def trajectoriesW(debut,fin,y0,nombrePoints,std,nameActivation,nbTirages,method,approxHessian=False):
    plt.title("Différentes trajectoires au voisinage d'un minimum")
    plt.xlabel("t")
    plt.ylabel("w")
    for nb in range(nbTirages):
        if(method=="normalGrad"):
            t_list,y_list = normalGrad(debut,fin,y0,nombrePoints,std,nameActivation,nb,approxHessian)
        elif(method=="expGrad"):
            t_list,y_list = expGrad(debut,fin,y0,nombrePoints,std,nameActivation,nb,approxHessian)
        elif(method=="expR"):
            t_list,y_list = expR(debut,fin,y0,nombrePoints,std,nameActivation,nb)
        elif(method=="normalR"):
            t_list,y_list = normalR(debut,fin,y0,nombrePoints,std,nameActivation,nb)
        plt.plot(t_list,y_list[:,0])
    plt.legend()
    plt.show()

def opti(y0,eta,std,eps,nameActivation,method,approxHessian=False,maxIter=2000):
    yt = cp.deepcopy(y0)
    iter=0
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        if(method=="normalGrad"):
            if(approxHessian):
                M = quasiHessian(yt,nameActivation)
            else:
                M = Hessian(yt,nameActivation)
            yt += -2*eta*np.dot(M,gradient)+2*std*np.sqrt(eta)*np.random.normal(size=(2,1))
        if(method=="expGrad"):
            if(approxHessian):
                M = quasiHessian(yt,nameActivation)
            else:
                M = Hessian(yt,nameActivation)
            yt += -eta/(np.linalg.norm(gradient))*np.dot(M,gradient)+std*np.sqrt(eta)*np.random.normal(size=(2,1))
        elif(method=="expR"):
            yt+=-eta*grad(yt,nameActivation)+std*np.sqrt(eta)*np.random.normal(size=(2,1))
        elif(method=="normalR"):
            yt+=-2*eta*R(yt,nameActivation)*grad(yt,nameActivation)+2*std*np.sqrt(eta)*np.random.normal(size=(2,1))
        gradient = grad(yt,nameActivation)
        iter+=1
    print(iter)
    return yt


nameActivation = "polyThree"
y0=np.zeros((2,1)); y0[0,0]=-0.1; y0[1,0]=0.8

debut=0
fin=2
nombrePoints=2000
time=np.linspace(debut,fin,nombrePoints)

nbTirages=10
approxHessian=False
method="expGrad"
#trajectoriesW(debut,fin,y0,nombrePoints,std,nameActivation,nbTirages,method,approxHessian)

eta=0.0001
eps=10**(-7)
maxIter=20000
std=0.1

#result = opti(y0,eta,std,eps,nameActivation,method,approxHessian,maxIter)
print(result)
