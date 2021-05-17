#include "test.h"

void testPolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int nbTirage, std::string const algo, double mu, double factor,double const Rlim, double const RMin,
 double const RMax, double const epsDiag, int const b, double const factorMin, double const power, double const alphaChap, double const eps, int maxIter, double epsNeight)
{
    Eigen::MatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyTwo";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    int i; unsigned seed;
    std::map<std::string,double> study;
    Eigen::VectorXd currentPoint(2);
    std::vector<Eigen::VectorXd> points(4);
    std::vector<double> proportions(4,0.0), distances(4,0.0), iters(4,0.0), backs(4,0.0);
    int numeroPoint, farMin=0, nonMin=0;

    points[0]=Eigen::VectorXd::Zero(2); points[1]=Eigen::VectorXd::Zero(2); points[2]=Eigen::VectorXd::Zero(2); points[3]=Eigen::VectorXd::Zero(2);
    points[0](0)=-2; points[0](1)=1; points[1](0)=2; points[1](1)=-1; points[2](0)=0; points[2](1)=-1; points[3](0)=0; points[3](1)=1;
    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        if(algo=="LM") {study = LM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter);}
        else if(algo=="LMF") {study = LMF(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax);}
        else if(algo=="LMUphill") {study = LMUphill(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b);}
        else if(algo=="LMPerso") {study = LMPerso(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b,epsDiag);}
        else if(algo=="LMJynian") {study = LMJynian(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,Rlim,RMin,RMax,factorMin,power,alphaChap);}

        if (study["finalGradient"]<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée" << std::endl; farMin++;}
            iters[numeroPoint]+=study["iter"];
            backs[numeroPoint]+=study["propBack"];
        }
        else
        {
            nonMin++;
            std::cout << "On n'est pas tombé sur un minimum" << std::endl;
        }

    }
    for(i=0;i<4;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i]; iters[i]/=proportions[i]; backs[i]/=proportions[i];}
        proportions[i]/=(double)nbTirage;
    }
    std::cout << "La proportion pour (-2,1): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (-2,1): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,1): " << iters[0]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (-2,1): " << backs[0] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,-1): " << proportions[1] << std::endl;
    std::cout << "La distance moyenne à (2,-1): " << distances[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,-1): " << iters[1]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (2,-1): " << backs[1] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,-1): " << proportions[2] << std::endl;
    std::cout << "La distance moyenne à (0,-1): " << distances[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[2]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (0,-1): " << backs[2] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,1): " << proportions[3] << std::endl;
    std::cout << "La distance moyenne à (0,1): " << distances[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,1): " << iters[3]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (0,1): " << backs[3] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << (double)(nonMin)/(double)nbTirage << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << (double)(farMin)/(double)nbTirage << std::endl;

}

void testPolyThree(std::string const& distribution, std::vector<double> const& supParameters, int nbTirage, std::string const algo, double mu, double factor,double const Rlim, double const RMin,
 double const RMax, double const epsDiag, int const b, double const factorMin, double const power, double const alphaChap, double const eps, int maxIter, double epsNeight)
{
    Eigen::MatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyThree";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    int i; unsigned seed;
    std::map<std::string,double> study;
    Eigen::VectorXd currentPoint(2);
    std::vector<Eigen::VectorXd> points(5);
    std::vector<double> proportions(5,0.0), distances(5,0.0), iters(5,0.0), backs(5,0.0);
    int numeroPoint, farMin=0, nonMin=0;


    points[0]=Eigen::VectorXd::Zero(2); points[1]=Eigen::VectorXd::Zero(2); points[2]=Eigen::VectorXd::Zero(2); points[3]=Eigen::VectorXd::Zero(2); points[4]=Eigen::VectorXd::Zero(2);
    points[0](0)=0; points[0](1)=-1; points[1](0)=2; points[1](1)=-1; points[2](0)=-2; points[2](1)=1; points[3](0)=-1; points[3](1)=0; points[4](0)=0; points[4](1)=1;
    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        if(algo=="LM") {study = LM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter);}
        else if(algo=="LMF") {study = LMF(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax);}
        else if(algo=="LMUphill") {study = LMUphill(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b);}
        else if(algo=="LMPerso") {study = LMPerso(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b,epsDiag);}
        else if(algo=="LMJynian") {study = LMJynian(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,Rlim,RMin,RMax,factorMin,power,alphaChap);}

        if (study["finalGradient"]<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint<0){farMin++;}
            iters[numeroPoint]+=study["iter"];
            backs[numeroPoint]+=study["propBack"];
        }
        else {nonMin++;}

    }
    for(i=0;i<5;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i]; iters[i]/=proportions[i]; backs[i]/=proportions[i];}
        proportions[i]/=(double)nbTirage;
    }
    std::cout << "La proportion pour (0,-1): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (0,-1): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[0]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (0,-1): " << backs[0] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,-1): " << proportions[1] << std::endl;
    std::cout << "La distance moyenne à (2,-1): " << distances[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,-1): " << iters[1]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (2,-1): " << backs[1] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (-2,1): " << proportions[2] << std::endl;
    std::cout << "La distance moyenne à (-2,1): " << distances[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,1): " << iters[2]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (-2,1): " << backs[2] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (-1,0): " << proportions[3] << std::endl;
    std::cout << "La distance moyenne à (-1,0): " << distances[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-1,0): " << iters[3]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (-1,0): " << backs[3] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,1): " << proportions[4] << std::endl;
    std::cout << "La distance moyenne à (0,1): " << distances[4] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,1): " << iters[4]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (0,1): " << backs[4] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << (double)(nonMin)/(double)nbTirage << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << (double)(farMin)/(double)nbTirage << std::endl;

}

void testCloche(std::string const& distribution, std::vector<double> const& supParameters, int nbTirage, std::string const algo, double mu, double factor,double const Rlim, double const RMin,
 double const RMax, double const epsDiag, int const b, double const factorMin,double const power,double const alphaChap, double const eps, int maxIter, double epsNeight)
{
    Eigen::MatrixXd X(1,3), Y(1,3);
    X(0,0)=0; X(0,1)=1; X(0,2)=2; Y(0,0)=1; Y(0,1)=0; Y(0,2)=1;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="cloche";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    int i; unsigned seed;
    std::map<std::string,double> study;
    Eigen::VectorXd currentPoint(2);
    std::vector<Eigen::VectorXd> points(2);
    std::vector<double> proportions(2,0.0), distances(2,0.0), iters(2,0.0), backs(2,0.0);
    int numeroPoint, nonMin=0, farMin=0;

    points[0]=Eigen::VectorXd::Zero(2); points[1]=Eigen::VectorXd::Zero(2);
    points[0](0)=0; points[0](1)=-std::sqrt(2*std::log(3.0/2.0)); points[1](0)=0; points[1](1)=std::sqrt(2*std::log(3.0/2.0));
    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        if(algo=="LM") {study = LM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter);}
        else if(algo=="LMF") {study = LMF(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax);}
        else if(algo=="LMUphill") {study = LMUphill(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b);}
        else if(algo=="LMPerso") {study = LMPerso(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b,epsDiag);}
        else if(algo=="LMJynian") {study = LMJynian(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,Rlim,RMin,RMax,factorMin,power,alphaChap);}

        if (study["finalGradient"]<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée" << std::endl; farMin++;}
            iters[numeroPoint]+=study["iter"];
            backs[numeroPoint]+=study["propBack"];
        }
        else
        {
            nonMin++;
            std::cout << "On n'est pas tombé sur un minimum" << std::endl;
        }
    }
    for(i=0;i<4;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i]; iters[i]/=proportions[i]; backs[i]/=proportions[i];}
        proportions[i]/=(double)nbTirage;
    }
    std::cout << "La proportion pour (0,-z0): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (0,-z0): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-z0): " << iters[0]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (0,-z0): " << backs[0] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,z0): " << proportions[1] << std::endl;
    std::cout << "La distance moyenne à (0,z0): " << distances[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,z0): " << iters[1]<< std::endl;
    std::cout << "La proportion de retours en arrière pour arriver à (0,z0): " << backs[1] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << (double)(nonMin)/(double)nbTirage << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << (double)(farMin)/(double)nbTirage << std::endl;


}

