from unicodedata import name
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random as random
import copy as cp

from scipy import stats
import seaborn as sns

random.seed(10)

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

def G(theta,thetaStar,gradient,k,seuilK=1):
    if(k<seuilK):
        r=k
    else:
        r=1/k**2
    result = np.sqrt(2*k)+np.sqrt(r)*(np.linalg.norm(gradient)/np.linalg.norm(theta-thetaStar))
    return result*np.eye(2)

def distance(theta,thetas,epsNeight=10**(-3)):
    nbMins = len(thetas)
    for k in range(nbMins):
        if(np.linalg.norm(thetas[k]-theta)<epsNeight):
            return k
    return -1

def compute_r(k,seuilK=1):
    if(k<seuilK):
        r=k
    else:
        r=1/k**2
    return r

def sigma(theta,nameActivation):
    w,b = theta[0,0],theta[1,0]
    sigma = np.zeros((2,2))
    sigma[0,0] = gp(w+b,nameActivation)*g(w+b,nameActivation)
    sigma[1,0] = sigma[0,0]-gp(b,nameActivation)*g(b,nameActivation)
    return 0.5*np.sign(gp(w+b,nameActivation)*g(w+b,nameActivation))*sigma

def SGD(theta0,eta,std,eps,nameActivation,maxIter):
    theta = cp.deepcopy(theta0)
    iter=1
    gradient = grad(theta,nameActivation)
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        theta += -eta*gradient + std*np.sqrt(eta)*np.random.normal()*np.dot(sigma(theta,nameActivation),np.random.normal(size=(2,1)))
        gradient = grad(theta,nameActivation)
        iter+=1
    return theta,iter

def GD_stabilisation(theta0,eta,std,eps,nameActivation,thetaStar,maxIter):
    theta = cp.deepcopy(theta0)
    iter=1
    gradient = grad(theta,nameActivation)
    k = np.linalg.norm(gradient)/np.linalg.norm(theta-thetaStar)
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        theta += -eta*gradient + std*np.sqrt(eta)*np.random.normal()*np.dot(G(theta,thetaStar,gradient,k),theta-thetaStar)
        k = np.maximum(k,np.linalg.norm(gradient)/np.linalg.norm(theta-thetaStar))
        gradient = grad(theta,nameActivation)
        iter+=1
    return theta,iter

def GD_stabilisation2(theta0,eta,std,eps,nameActivation,thetaStar,maxIter):
    theta = cp.deepcopy(theta0)
    iter=1
    gradient = grad(theta,nameActivation)
    k = np.linalg.norm(gradient)/np.linalg.norm(theta-thetaStar)
    r=compute_r(k)
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        theta += -eta*gradient + std*np.sqrt(eta)*np.random.normal()*(np.sqrt(k)+np.sqrt(r))*(theta-thetaStar)
        k = np.maximum(k,np.linalg.norm(gradient)/np.linalg.norm(theta-thetaStar))
        r=compute_r(k)
        gradient = grad(theta,nameActivation)
        iter+=1
    return theta,iter

def GD1(theta0,eta,std,eps,nameActivation,maxIter):
    theta = cp.deepcopy(theta0)
    iter=1
    gradient = grad(theta,nameActivation)
    r=1
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        theta += -eta*gradient + std*np.sqrt(eta)*np.random.normal()*(np.sqrt(2)+np.sqrt(r))*gradient
        #theta += -eta*gradient + std*np.sqrt(eta)*(np.sqrt(2)+np.sqrt(r))*np.dot(np.dot(np.abs(gradient),np.ones((1,2))),np.random.normal(size=(2,1)))
        gradient = grad(theta,nameActivation)
        iter+=1
    return theta,iter

def HeunGD1(theta0,eta,std,eps,nameActivation,maxIter):
    theta = cp.deepcopy(theta0)
    iter=1
    gradient = grad(theta,nameActivation)
    r=1
    thetaInter = theta-eta*gradient+std*(np.sqrt(2)+r)*np.sqrt(eta)*np.random.normal()*gradient
    gradientInter = grad(thetaInter,nameActivation)
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        theta += -eta/2*(gradient+gradientInter) +np.sqrt(eta)/2*(np.sqrt(2)+r)*np.random.normal()*(gradient+gradientInter)
        gradient = grad(theta,nameActivation)
        thetaInter = theta-eta*gradient+std*(np.sqrt(2)+r)*np.sqrt(eta)*np.random.normal()*gradient
        gradientInter = grad(thetaInter,nameActivation)
        iter+=1
    return theta
    


def Momentum(theta0,eta,beta,eps,nameActivation,maxIter):
    theta = cp.deepcopy(theta0); v=np.zeros((2,1))
    gradient = grad(theta,nameActivation)
    iter=1
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        v += -beta*v+beta*gradient
        theta += -eta*v
        gradient = grad(theta,nameActivation)
        iter+=1
    return theta,iter

def MomentumEuler(theta0,eta,beta,eps,nameActivation,maxIter):
    theta = cp.deepcopy(theta0); v=np.zeros((2,1))
    gradient = grad(theta,nameActivation)
    iter=1
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        v,vPrec = v-beta*v+beta*gradient,v
        theta += -eta*vPrec
        gradient = grad(theta,nameActivation)
        iter+=1
    return theta,iter


def MomentumPers(theta0,std,eta,beta,eps,nameActivation,maxIter):
    theta = cp.deepcopy(theta0); v=np.zeros((2,1))
    gradient = grad(theta,nameActivation)
    r=0.5
    iter=1
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        alea = np.random.normal()
        v,vPrec = v-beta*v+beta*gradient+std*(1+r)*np.sqrt(eta)*alea*v,v
        theta += -eta*vPrec+std*(1+r)*np.sqrt(beta)*alea*(gradient-vPrec)
        gradient = grad(theta,nameActivation)
        iter+=1
    return theta,iter

def MomentumPers2(theta0,std,eta,beta,eps,nameActivation,maxIter):
    theta = cp.deepcopy(theta0); v=np.zeros((2,1))
    gradient = grad(theta,nameActivation)
    r=0.5
    iter=1
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        v += -beta*v+beta*gradient+std*(1+r)*np.sqrt(eta)*np.random.normal()*v
        theta += -eta*v+std*(1+r)*np.sqrt(beta)*np.random.normal()*(gradient-v)
        gradient = grad(theta,nameActivation)
        iter+=1
    return theta,iter

def ERIto(theta0,eta,seuil,eps,eps1,nameActivation,maxIter):
    theta = cp.deepcopy(theta0)
    iter=1
    gradient = grad(theta,nameActivation)
    r=1
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        gradientInter = grad(theta-eta/2*gradient,nameActivation)
        erreur = (eta/2*np.linalg.norm(gradient-gradientInter,1))/seuil
        if(erreur>1):
            eta*=0.9/np.sqrt(erreur)
        else:
            theta -= eta*gradientInter
            eta *= 0.9/np.sqrt(erreur)
            gradient = grad(theta,nameActivation)
        iter+=1
        print(eta)
    return theta,iter



def distance(theta,thetas,epsNeight=10**(-3)):
    nbMins = len(thetas)
    for k in range(nbMins):
        if(np.linalg.norm(thetas[k]-theta)<epsNeight):
            return k
    return -1

def statistics(theta0,eta,beta,opti,nameActivation,eps=10**(-7),maxIter=2000,epsNeight=10**(-3),nbTirages=10000):
    thetas=[np.array([[-2],[1]]), np.array([[2],[-1]]), np.array([[0],[-1]]), np.array([[0],[1]]), np.array([[0],[0]]), np.array([[-1],[0]]), np.array([[1],[0]]), np.array([[-1],[1]]), np.array([[1],[-1]])]
    labels=["(-2,1)", "(2,-1)", "(0,-1)","(0,1)","(0,0)","(-1,0)","(1,0)","(-1,1)","(1,-1)"]
    props=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]; itersMoy=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]; iters=[[],[],[],[],[],[],[],[],[]]
    div=0; farMin=0

    for k in range(nbTirages):
        theta,iter = opti(theta0,eta,1,eps,nameActivation,maxIter)
        #theta,iter = opti(theta0,eta,beta,1,eps,nameActivation,maxIter)
        if(np.isnan(theta.any()) or np.isinf(theta.any()) or np.linalg.norm(theta)>1000):
            div+=1
        else:
            numero = distance(theta,thetas,epsNeight)
            if(numero==-1):
                farMin+=1
                print(theta)
            else:
                props[numero]+=1
                iters[numero].append(iter)
                itersMoy[numero]+=iter
    for k in range(len(thetas)):
        if(props[k]!=0):
            itersMoy[k]/=props[k]
        props[k]/=nbTirages
    div/=nbTirages; farMin/=nbTirages
    for k in range(len(thetas)):
        print("Proportions de ", labels[k], ": ", props[k])
        print("Moyenne d'itérations pour ", labels[k], ": ", itersMoy[k])
    print("Proportions de divergence: ", div)
    print("Proportions de farMin: ", farMin)

    fig,axes = plt.subplots(3,3,sharex=True,figsize=(10,5))
    axes = axes.flatten()
    for k in range(len(thetas)):
        axes[k].set_title("Point "+labels[k])
        sns.histplot(iters[k], stat='density',ax=axes[k])
    plt.show()



nameActivation = "polyTwo"
theta0=np.zeros((2,1)); theta0[0,0]=-0.5; theta0[1,0]=0.5
thetaStar=np.zeros((2,1)); thetaStar[0,0]=0; thetaStar[1,0]=1

eta=10**(-1); seuil=0.01
beta=1-0.9
eps=10**(-7); eps1=10**(-2)
maxIter=2000
std=1

#result = GD_stabilisation2(theta0,eta,std,eps,nameActivation,thetaStar,maxIter)
#result,iter = GD1(theta0,eta,0,eps,nameActivation,maxIter)
#result,iter = MomentumEuler(theta0,eta,beta,eps,nameActivation,maxIter)
result,iter = ERIto(theta0,eta,seuil,eps,eps1,nameActivation,maxIter)
print(result); print(iter)

#statistics(theta0,eta,beta,HeunGD1,nameActivation,eps,maxIter)