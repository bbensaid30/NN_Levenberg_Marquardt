from os import name
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.integrate import odeint
import time
import random
random.seed(10)

"""
Script sur la résolution du sytème pour GD pour les exemples par Euler implicite (optimisation)
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

def Newton_Raphson(theta_n,grad_n,x0,eta,nameActivation,epsSolver=10**(-8)):
    I = np.identity(2)
    x=x0
    G = theta_n-0.5*eta*(grad_n+grad(x,nameActivation))-x
    iter=0; maxIter=1000
    while(np.linalg.norm(G)>epsSolver and iter<maxIter):
        try:
            delta = np.linalg.solve(0.5*eta*Hessian(x,nameActivation)+I,G)
        except np.linalg.LinAlgError:
            return x
        x+=delta
        G = theta_n-eta/2*(grad_n+grad(x,nameActivation))-x
        iter+=1
    #print(iter)
    return x

def crankNicholson_implicite_opti(theta0,eta,nameActivation,solver,eps=10**(-7),maxIter=2000):
    theta = theta0
    gradient = grad(theta,nameActivation)
    iter=0
    while(np.linalg.norm(gradient)>eps and iter<maxIter):
        theta = solver(theta,gradient,theta,eta,nameActivation,eps*eta)
        gradient = grad(theta,nameActivation)
        iter+=1
    return theta,iter

def distance(theta,thetas,epsNeight=10**(-3)):
    nbMins = len(thetas)
    for k in range(nbMins):
        if(np.linalg.norm(thetas[k]-theta)<epsNeight):
            return k
    return -1

def statistics(eta,nameActivation,solver,eps=10**(-7),maxIter=2000,epsNeight=10**(-3),nbTirages=10000,intervals=(-3,3)):
    nonMin=0; farMin=0
    if(nameActivation=="polyTwo" or nameActivation=="polyThree" or nameActivation=="polyFour"):
        nbMins=4
        thetas=[np.array([[-2],[1]]), np.array([[2],[-1]]), np.array([[0],[-1]]), np.array([[0],[1]])]
        labels=("(-2,1)", "(2,-1)", "(0,-1)","(0,1)")
        props=[0.0,0.0,0.0,0.0]; distances=[0.0,0.0,0.0,0.0]; iters=[0.0,0.0,0.0,0.0]
        ws = [[],[],[],[]]; bs = [[],[],[],[]]
        colors=["blue","orange","gold","magenta"]
    elif nameActivation == "polyFive":
        nbMins=6
        thetas=[np.array([[2],[1]]), np.array([[0],[-1]]), np.array([[-2],[3]]), np.array([[0],[3]]), np.array([[-4],[3]]), np.array([[4],[-1]])]
        labels=("(2,1)", "(0,-1)", "(-2,3)","(0,3)", "(-4,3)", "(4,-1)")
        props=[0.0,0.0,0.0,0.0,0.0,0.0]; distances=[0.0,0.0,0.0,0.0,0.0,0.0]; iters=[0.0,0.0,0.0,0.0,0.0,0.0]
        ws = [[],[],[],[],[],[]]; bs = [[],[],[],[],[],[]]
        colors=["blue","orange","gold","red","magenta","black"]
    elif nameActivation == "polyEight":
        nbMins=6
        thetas=[np.array([[0],[0]]), np.array([[2],[0]]), np.array([[-2],[2]]), np.array([[1],[0]]), np.array([[-1],[1]]), np.array([[0],[2]])]
        labels=("(0,0)", "(2,0)", "(-2,2)","(1,0)", "(-1,1)", "(0,2)")
        props=[0.0,0.0,0.0,0.0,0.0,0.0]; distances=[0.0,0.0,0.0,0.0,0.0,0.0]; iters=[0.0,0.0,0.0,0.0,0.0,0.0]
        ws = [[],[],[],[],[],[]]; bs = [[],[],[],[],[],[]]
        colors=["red","orange","gold","blue","magenta","black"]
    
    start = time.perf_counter()
    for nb in range(nbTirages):
        theta0 = np.random.uniform(intervals[0],intervals[1],size=(2,1))
        theta_init = np.copy(theta0)
        theta,iter = crankNicholson_implicite_opti(theta0,eta,nameActivation,solver,eps)
        if(np.linalg.norm(grad(theta,nameActivation))<eps):
            nMin = distance(theta,thetas,epsNeight)
            if(nMin<0):
                print("On n'est pas assez proche du min: ", np.linalg.norm(thetas[nMin]-theta))
                print("On est au tirage: ", nb)
                farMin+=1
            else:
                ws[nMin].append(theta_init[0,0]); bs[nMin].append(theta_init[1,0])
                props[nMin]+=1
                distances[nMin] = np.linalg.norm(thetas[nMin]-theta)
                iters[nMin]+=iter
        else:
            print("La condition sur le gradient n'est pas respectée")
            print("On est au tirage: ", nb)
            nonMin+=1
    end = time.perf_counter()
    
    fig = plt.figure(figsize=(10,10))
    axes = fig.add_subplot(111)
    axes.set_frame_on(True)
    axes.add_artist(patches.Rectangle((intervals[0],intervals[0]),np.abs(intervals[1]-intervals[0]),np.abs(intervals[1]-intervals[0]),color="red",fill=False))

    axes.set_xlim(intervals[0],intervals[1])
    axes.set_xlabel("w")
    axes.set_ylim(intervals[0],intervals[1])
    axes.set_ylabel("b")
    axes.set_title("Ensemble des points d'initialisation convergeant vers un certain miminum")

    for min in range(nbMins):
        distances[min]/=props[min]
        iters[min]/=props[min]
        props[min]/=nbTirages
        print("Proportion pour le point ", labels[min], " : ", props[min])
        print("Distance moyenne pour le point ", labels[min], " : ", distances[min])
        print("Nombre d'itérations moyenne pour le point ", labels[min], " : ", iters[min])
        axes.scatter(ws[min], bs[min], color=colors[min], label=labels[min])
    
    print("Proportions de fois où la condition sur le gradient n'est pas respectée: ", nonMin/nbTirages)
    print("Proportions de fois où on n'est pas assez proche du min même si la condition sur le gradient est respectée: ", farMin/nbTirages)

    print("Temps: ", end-start)
    
    axes.legend()
    fig.show(); plt.show()
    


theta0=np.array([[0.5],[0.5]])
eta=0.1
nameActivation="polyEight"
solver = Newton_Raphson
eps=10**(-7)
maxIter=20000
epsNeight=10**(-3)
nbTirages=10000
intervals=(-3,3)
#theta = crankNicholson_implicite_opti(theta0,eta,nameActivation,solver,eps,maxIter); print(theta)

statistics(eta,nameActivation,solver,eps,maxIter,epsNeight,nbTirages,intervals)