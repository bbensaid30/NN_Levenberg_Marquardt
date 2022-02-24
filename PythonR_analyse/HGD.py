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

#H(v,theta)=0.5*|v|^2+R(theta) par Euler symplectique
def HGD_EulerA(y0,eta,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0)
    v=np.zeros((2,1))
    iter=0
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        yt += eta*v
        gradient = grad(yt,nameActivation)
        v -= eta* gradient
        iter+=1
        #print(0.5*np.linalg.norm(v)**2+R(yt,nameActivation))
    print(iter)
    return yt


#H(v,theta)=0.5*|v|^2+R(theta) par un Verlet à un pas
def HGD_Verlet(y0,v0,eta,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0)
    v = cp.deepcopy(v0)
    iter=0
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        yt += eta*v-(eta**2)/2*gradient
        gradientPrec,gradient = gradient,grad(yt,nameActivation)
        v -= (eta/2)*(gradientPrec+gradient) 
        iter+=1
        #print(0.5*np.linalg.norm(v)**2+R(yt,nameActivation))
    print(iter)
    print(v)
    return yt

def HGDT_Verlet(y0,v0,eta,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0)
    v = cp.deepcopy(v0)
    iter=0; t=1
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        yt += eta*v-(eta**2)/2*(gradient/t)
        gradientPrec,gradient = gradient,grad(yt,nameActivation)
        v -= (eta/2)*(gradientPrec/t+gradient/(t+eta)) 
        iter+=1; t+=eta
        print(0.5*np.linalg.norm(v)**2+R(yt,nameActivation)/t)
    print(iter)
    print(v)
    return yt

def HGDT_Euler(y0,v0,eta,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0)
    v = cp.deepcopy(v0)
    iter=0; t=1
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        yt += eta*v
        v -= eta*(gradient/t)
        gradient = grad(yt,nameActivation)
        iter+=1; t+=eta
        #print(0.5*np.linalg.norm(v)**2+R(yt,nameActivation)/t)
    print(iter)
    print(v)
    return yt

def Langevin(y0,v0,pas,beta,u,std,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0)
    v = cp.deepcopy(v0)
    iter=0
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        a = beta*v+u*gradient
        yt += pas*v-pas**2/2*a
        gradient = grad(yt,nameActivation)
        v = (v-pas/2*(a+u*gradient))/(1+pas*beta/2)
        iter+=1
    print(iter)
    print(v)
    return yt

def Momentum(y0,v0,eta,beta,eps,nameActivation,maxIter=2000):
    theta = cp.deepcopy(y0); v=cp.deepcopy(v0)
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

def MomentumEuler(y0,v0,eta,beta,eps,nameActivation,maxIter=2000):
    theta = cp.deepcopy(y0); v=cp.deepcopy(v0)
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

def Momentum_Verlet(y0,v0,eta,beta,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0)
    v = cp.deepcopy(v0)
    beta_bar = beta/eta
    E = 0.5*np.linalg.norm(v)**2+beta_bar*R(yt,nameActivation)
    iter=1;count=0
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        a = v+gradient
        yt += eta*v-(eta*beta/2)*a
        gradient,gradientPrec = grad(yt,nameActivation),gradient
        v = ((1-beta/2)*v-beta/2*(gradient+gradientPrec))/(1+beta/2)
        E,EPrec = 0.5*np.linalg.norm(v)**2+beta_bar*R(yt,nameActivation),E
        if(E-EPrec>0):
            count+=1
        iter+=1
    print(iter)
    print(count/iter)
    return yt


def MomentumBis(y0,v0,eta,beta,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0)
    v = cp.deepcopy(v0)
    E = 0.5*np.linalg.norm(v)**2+R(yt,nameActivation)
    iter=1;count=0
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        v = -eta*np.linalg.norm(gradient)**2*v-eta*gradient
        yt += eta*v
        E,EPrec = 0.5*np.linalg.norm(v)**2+R(yt,nameActivation),E
        gradient = grad(yt,nameActivation)
        if(E-EPrec>0):
            count+=1
        iter+=1
    print(iter)
    print(count/iter)
    return yt

def MomentumBis_Verlet(y0,v0,eta,beta,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0)
    v = cp.deepcopy(v0)
    E = 0.5*np.linalg.norm(v)**2+R(yt,nameActivation)
    iter=1;count=0
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        a = np.linalg.norm(gradient)**2*v+gradient
        yt += eta*v-(eta**2/2)*a
        gradient = grad(yt,nameActivation)
        v = (v-eta/2*(a+gradient))/(1+eta/2*np.linalg.norm(gradient)**2)
        E,EPrec = 0.5*np.linalg.norm(v)**2+R(yt,nameActivation),E
        if(E-EPrec>0):
            count+=1
        iter+=1
    print(iter)
    print(count/iter)
    return yt

def Lagrangian_Verlet(y0,v0,eta,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0)
    v = cp.deepcopy(v0)
    E = 0.5*np.linalg.norm(v)**2+R(yt,nameActivation)
    iter=1;t=1; count=0
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        a = (4/t)*v+t*gradient
        yt += eta*v-(eta**2/2)*a
        gradient,gradientPrec = grad(yt,nameActivation),gradient
        v = (v-eta/2*(a+(t+eta)*gradient))/(1+2*eta/(t+eta))
        t+=eta
        E,EPrec = 0.5*np.linalg.norm(v)**2+R(yt,nameActivation),E
        if(E-EPrec>0):
            count+=1
        iter+=1
    print(iter)
    print(count/iter)
    return yt

def LagrangianBis_Verlet(y0,v0,eta,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0)
    v = cp.deepcopy(v0)
    E = 0.5*np.linalg.norm(v)**2+R(yt,nameActivation)
    iter=1;t=1; count=0
    gradient = grad(yt,nameActivation)
    while (np.linalg.norm(gradient)>eps and iter<maxIter):
        a = (4/t)*v+gradient
        yt += eta*v-(eta**2/2)*a
        gradient,gradientPrec = grad(yt,nameActivation),gradient
        v = (v-eta/2*(a+gradient))/(1+2*eta/(t+eta))
        t+=eta
        E,EPrec = 0.5*np.linalg.norm(v)**2+R(yt,nameActivation),E
        if(E-EPrec>0):
            count+=1
        iter+=1
    print(iter)
    print(count/iter)
    return yt

#H(v,theta) = R(theta)+R(v) par Euler symplectique
def PGD(y0,v0,eta,eps,nameActivation,maxIter=2000):
    yt = cp.deepcopy(y0); v=cp.deepcopy(v0)
    iter=0
    gradientY = grad(yt,nameActivation)
    gradientV = grad(v,nameActivation)
    while (np.linalg.norm(gradientY)>eps and np.linalg.norm(gradientV)>eps and iter<maxIter):
        yt += eta*gradientV
        gradientY = grad(yt,nameActivation)
        v -= eta*gradientY
        gradientV = grad(v,nameActivation)
        iter+=1
        #print(R(yt,nameActivation)+R(v,nameActivation))
    print(iter)
    if(R(yt,nameActivation)<R(v,nameActivation)):
        return yt
    else:
        return v

def F(theta,thetan,gradientVn,eta,nameActivation):
    return theta-thetan-eta*R(theta,nameActivation)*gradientVn

def JF(theta,gradientVn,eta,nameActivation):
    return np.eye(2)-eta*np.dot(gradientVn,np.transpose(grad(theta,nameActivation)))

def Newton_Raphson(thetan,gradientVn,eta,nameActivation,epsNewton,maxIterNewton=2000):
    theta = cp.deepcopy(thetan)
    value = F(theta,thetan,gradientVn,eta,nameActivation)
    iter=0
    while (np.linalg.norm(value)>epsNewton and iter<maxIterNewton):
        theta -= np.linalg.solve(JF(theta,gradientVn,eta,nameActivation),value)
        value = F(theta,thetan,gradientVn,eta,nameActivation)
        iter+=1
    return theta

#H(v,theta)=R(theta)*R(v)
def prodEulerA(y0,v0,eta,eps,nameActivation,maxIter=2000):
    theta=cp.deepcopy(y0); v=cp.deepcopy(v0)
    iter=0
    gradientTheta = grad(theta,nameActivation)
    gradientV = grad(v,nameActivation)
    while (np.linalg.norm(gradientTheta)>eps and np.linalg.norm(gradientV)>eps and iter<maxIter):
        theta = Newton_Raphson(theta,gradientV,eta,nameActivation,eps)
        gradientTheta = grad(theta,nameActivation)
        v -= eta*R(v,nameActivation)*gradientTheta
        gradientV = grad(v,nameActivation)
        iter+=1
        print(R(theta,nameActivation)*R(v,nameActivation))
    print(iter)
    if(R(theta,nameActivation)<R(v,nameActivation)):
        return theta
    else:
        return v


nameActivation = "polyThree"
y0=np.zeros((2,1)); y0[0,0]=0.5; y0[1,0]=-2.9
v0=np.zeros((2,1)); v0[0,0]=0; v0[1,0]=0
eta=0.1; pas=0.5
beta=1-0.9; u=1
std=0
eps=10**(-7)
maxIter=100000
#result = Langevin(y0,v0,pas,beta,u,std,eps,nameActivation,maxIter)
#result = MomentumEuler(y0,v0,eta,beta,eps,nameActivation,maxIter)
#result = Momentum(y0,v0,eta,beta,eps,nameActivation,maxIter)
result = Momentum_Verlet(y0,v0,eta,beta,eps,nameActivation,maxIter)
#result = MomentumBis_Verlet(y0,v0,eta,beta,eps,nameActivation,maxIter)
#result = LagrangianBis_Verlet(y0,v0,eta,eps,nameActivation,maxIter)
#result = HGDT_Verlet(y0,v0,eta,eps,nameActivation,maxIter)
#result = PGD(y0,v0,eta,eps,nameActivation,maxIter)
#result = prodEulerA(y0,v0,eta,eps,nameActivation,maxIter)
print(result)