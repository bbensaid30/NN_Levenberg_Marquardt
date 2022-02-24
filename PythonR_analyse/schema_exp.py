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

def Momentum(y0,eta,beta,eps,nameActivation,maxIter=2000):
    theta = cp.deepcopy(y0); v=np.zeros((2,1))
    gradient = grad(theta,nameActivation)
    beta_bar = beta/eta
    E = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation)
    iter=1;count=0
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        v += -beta*v+beta*gradient
        theta += -eta*v
        E,EPrec = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation),E
        gradient = grad(theta,nameActivation)
        iter+=1
        if(E-EPrec>0):
            count+=1
    print(iter)
    print(count/iter)
    return theta

def MomentumEuler(y0,eta,beta,eps,nameActivation,maxIter=2000):
    theta = cp.deepcopy(y0); v=np.zeros((2,1))
    gradient = grad(theta,nameActivation)
    beta_bar = beta/eta
    E = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation)
    iter=1;count=0
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        v,vPrec = v-beta*v+beta*gradient,v
        theta += -eta*vPrec
        E,EPrec = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation),E
        gradient = grad(theta,nameActivation)
        iter+=1
        if(E-EPrec>0):
            count+=1
    print(iter)
    print(count/iter)
    return theta

def Momentum_Verlet(theta0,eta,beta,eps,nameActivation,maxIter=2000):
    theta = cp.deepcopy(theta0); v=np.zeros((2,1))
    beta_bar = beta/eta
    E = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation)
    iter=1;count=0
    gradient = grad(theta,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        a = v+gradient
        theta += eta*v-(eta*beta/2)*a
        gradient,gradientPrec = grad(theta,nameActivation),gradient
        v = ((1-beta/2)*v-beta/2*(gradient+gradientPrec))/(1+beta/2)
        E,EPrec = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation),E
        if(E-EPrec>0):
            count+=1
        iter+=1
    print(iter)
    print(count/iter)
    return theta

def IFEuler(theta0,eta,beta,nameActivation,eps,maxIter=2000):
    iter=1
    Theta=cp.deepcopy(theta0); V = np.zeros((2,1))
    t=0; beta_bar=beta/eta
    gradient = grad(Theta,nameActivation)
    E = 0.5*np.linalg.norm(V)**2+beta_bar*R(Theta,nameActivation)
    iter=1;count=0
    termExp = np.exp(beta_bar*t)
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        V -= beta*termExp*gradient
        Theta += eta*(termExp-1)*gradient
        t+=eta
        termExp=np.exp(beta_bar*t)
        theta = Theta-((1/termExp-1)/beta_bar)*V; v = 1/termExp*V
        E,EPrec = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation),E
        if(E-EPrec>0):
            count+=1
        gradient = grad(theta,nameActivation)
        iter+=1
    print(iter)
    print(count/iter)
    return theta

def IFBisEuler(theta0,eta,nameActivation,eps,maxIter=2000):
    iter=1
    Theta=cp.deepcopy(theta0); V = np.zeros((2,1))
    t=0; beta_bar=beta/eta
    gradient = grad(Theta,nameActivation)
    E = 0.5*np.linalg.norm(V)**2+beta_bar*R(Theta,nameActivation)
    iter=1;count=0
    termExp = 0
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        V -= eta*t**3*gradient
        Theta += eta*termExp*(t**3-1)*gradient
        t+=eta
        termExp=(t-1)/(3*np.log(t))
        theta = Theta-termExp*(t**(-3)-1)*V; v=t**(-3)*V
        E,EPrec = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation),E
        if(E-EPrec>0):
            count+=1
        gradient = grad(theta,nameActivation)
        iter+=1
    print(iter)
    print(count/iter)
    return theta

def IFBisLEuler(theta0,eta,nameActivation,eps,maxIter=2000):
    iter=1
    Theta=cp.deepcopy(theta0); V = np.zeros((2,1))
    t=0; beta_bar=beta/eta
    gradient = grad(Theta,nameActivation)
    E = 0.5*np.linalg.norm(V)**2+beta_bar*R(Theta,nameActivation)
    iter=1;count=0
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        V -= 3*eta/(1+t)*V+eta*gradient
        Theta += (3*t)/(2*(1+t))*V+t/2*gradient
        t+=eta
        theta = Theta+t/2*gradient; v=V
        E,EPrec = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation),E
        if(E-EPrec>0):
            count+=1
        gradient = grad(theta,nameActivation)
        iter+=1
    print(iter)
    print(count/iter)
    return theta

def IFBisL2Euler(theta0,eta,nameActivation,eps,maxIter=2000):
    iter=1
    theta=cp.deepcopy(theta0); V = np.zeros((2,1)); v=V
    t=0; beta_bar=beta/eta
    gradient = grad(theta,nameActivation)
    E = 0.5*np.linalg.norm(V)**2+beta_bar*R(theta,nameActivation)
    iter=1;count=0
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        V -= t**3*gradient
        theta += v
        t+=eta
        v = V/t**3
        E,EPrec = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation),E
        if(E-EPrec>0):
            count+=1
        gradient = grad(theta,nameActivation)
        iter+=1
    print(iter)
    print(count/iter)
    return theta


def EDTEuler(theta0,eta,beta,nameActivation,eps,maxIter=2000):
    iter=1
    theta=cp.deepcopy(theta0); v = np.zeros((2,1))
    t=0; beta_bar=beta/eta
    gradient = grad(theta,nameActivation)
    E = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation)
    iter=1;count=0
    termExp = np.exp(-beta)
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        v,vPrec = termExp*v+eta*termExp*gradient,v
        theta -= (termExp-1)/beta_bar*v+eta*(termExp-1)/beta_bar*gradient
        E,EPrec = 0.5*np.linalg.norm(v)**2+beta_bar*R(theta,nameActivation),E
        if(E-EPrec>0):
            count+=1
        gradient = grad(theta,nameActivation)
        iter+=1
    print(iter)
    print(count/iter)
    return theta

nameActivation = "polyThree"
theta0=np.zeros((2,1)); theta0[0,0]=0.5; theta0[1,0]=-2.9
eta=0.1
beta=1-0.9
eps=10**(-7)
maxIter=100000

#result = Momentum(theta0,eta,beta,eps,nameActivation,maxIter)
result = Momentum_Verlet(theta0,eta,beta,eps,nameActivation,maxIter)
#result = IFEuler(theta0,eta,beta,nameActivation,eps,maxIter)
#result = IFBisL2Euler(theta0,eta,nameActivation,eps,maxIter)
#result = EDTEuler(theta0,eta,beta,nameActivation,eps,maxIter)
print(result)