#include <iostream>
#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include <random>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"
#include <EigenRand/EigenRand>

#include <omp.h>

#include "init.h"
#include "data.h"
#include "training.h"
#include "test.h"
#include "tirage.h"

int main()
{
    omp_set_num_threads(omp_get_num_procs());


    //Paramètres généraux
    std::string const distribution="uniform";
    std::vector<double> const supParameters={-3,3};
    int const tirageMin=0;
    int const nbTirages=10000;
    std::string const famille_algo="Perso";
    std::string const algo="Momentum_Em";
    Sdouble const eps=std::pow(10,-7);
    int const maxIter=200000;
    Sdouble const epsNeight=std::pow(10,-2);

    //Paramètres des méthodes LM
    Sdouble mu=100, factor=10, RMin=0.25, RMax=0.75, epsDiag=std::pow(10,-16), Rlim=std::pow(10,-4), factorMin=std::pow(10,-8), power=1.0, alphaChap=1.1, alpha=0.75, pas=0.1;
    Sdouble tau=1, beta=2.0, gamma=3.0;
    int const b=1, p=3;

    Sdouble learning_rate=0.1;
    Sdouble clip=1/learning_rate;
    Sdouble seuil=0.1;
    Sdouble beta1 = 1-0.9;
    Sdouble beta2 = 1-0.999;
    int const batch_size = 2;

    std::string const setHyperparameters = "1";
    test_PolyThree(distribution,supParameters,nbTirages,famille_algo,algo,learning_rate,clip,seuil,beta1,beta2,batch_size,mu,factor,Rlim,RMin,RMax,
    epsDiag,b,factorMin,power,alphaChap,alpha,pas,eps,maxIter,epsNeight,true,false,true,setHyperparameters);

    return 0;
}
