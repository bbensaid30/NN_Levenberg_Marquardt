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

#include "test.h"
#include "propagation.h"

#include "init.h"
#include "data.h"
#include "training.h"
#include "study_base.h"
#include "study_graphics.h"
#include "utilities.h"

int main()
{
    // Charger les données
    std::vector<Eigen::SMatrixXd> dataTrain(2);
    std::vector<Eigen::SMatrixXd> dataTrainTest(4);
    int const nbPoints=40; Sdouble const percTrain=1; bool const reproductible=true;
    dataTrain = sineWave(nbPoints);
    dataTrainTest = trainTestData(dataTrain,percTrain,reproductible);

    // Architecture
    int const n0=dataTrainTest[0].rows(), nL=dataTrainTest[1].rows();
    int N=0;
    int const L=2;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    int l;
//    for(int l=0;l<L;l++)
//    {
//        nbNeurons[l]=1;
//        activations[l]="sigmoid";
//    }
//    nbNeurons[0]=1;
//    nbNeurons[L]=1;
//    if(L>=2){activations[L-1]="linear";}
    nbNeurons[0]=1;
    nbNeurons[1]=5;
    nbNeurons[2]=1;
    activations[0]="reLU";
    activations[1]="linear";
    for(l=0;l<L;l++)
    {
       N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    // Paramètres d'initialisation
    std::string const generator="uniform";
    std::vector<double> supParameters = {-10,10};

    // Paramètres d'entraînement
    std::string const algo="LMF";
    Sdouble eps=std::pow(10,-7);
    Sdouble mu=10, factor=10, RMin=0.25, RMax=0.75, epsDiag=std::pow(10,-16), Rlim=std::pow(10,-4), factorMin=std::pow(10,-8), power=1.0, alphaChap=1.1, alpha=0.75, pas=0.1;
    Sdouble tau=1, beta=2.0, gamma=3.0, sigma=0.1, radiusBall=std::pow(10,0);
    std::string const norm="2";
    int const b=1, p=3;
    int const tirageDisplay = 100, tirageMin=0;
    int const maxIter=2000; Sdouble const epsClose=std::pow(10,-3);
    int const nbTirages=10000, nbDichotomie=std::pow(2,4);
    std::string const strategy="CostSd"; Sdouble const flat=0.2;


    std::string const folder="sineWave/P=40|L=2";
    std::string const activationString="reLUl";
    std::string const fileExtension = "5-1";
    minsRecord(dataTrainTest,L,nbNeurons,globalIndices,activations,supParameters,generator,algo,nbTirages,eps,maxIter,mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,
    tau,beta,gamma,p,sigma,norm,radiusBall,tirageMin,tirageDisplay,folder,fileExtension);
    //denombrementMinsPost(dataTrainTest,L,nbNeurons,globalIndices,activations,algo,epsClose,nbDichotomie,eps,tirageDisplay,strategy,flat,folder,fileExtension);


    //double const x=0.5, aGraphic=0.5, bGraphic=7, pasGraphic=0.1;
    //nbMinsFlats(dataTrainTest,L,nbNeurons,globalIndices,activations,algo,epsClose,nbDichotomie,eps,folder,fileExtension,x,aGraphic,bGraphic,pasGraphic,strategy);

    //Shaman::displayUnstableBranches();

    return 0;
}
