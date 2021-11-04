#include "testSGD.h"

void testSGD_PolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& algo, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight)
{
    Eigen::SMatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);

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
    std::map<std::string,Sdouble> study;
    Eigen::SVectorXd currentPoint(2);
    std::vector<Eigen::SVectorXd> points(4);
    std::vector<Sdouble> proportions(4,0.0), distances(4,0.0), iters(4,0.0);
    int numeroPoint, farMin=0, nonMin=0;

    points[0]=Eigen::SVectorXd::Zero(2); points[1]=Eigen::SVectorXd::Zero(2); points[2]=Eigen::SVectorXd::Zero(2); points[3]=Eigen::SVectorXd::Zero(2);
    points[0](0)=-2; points[0](1)=1; points[1](0)=2; points[1](1)=-1; points[2](0)=0; points[2](1)=-1; points[3](0)=0; points[3](1)=1;
    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        study = train_SGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",algo,learning_rate,batch_size,beta1,beta2,eps,maxIter);
        std::cout << i << std::endl;
        if (std::abs(study["moyFinalGradient"].error)>eps)
        {
            std::cout << i << ": " << study["moyFinalGradient"].digits() << std::endl;
        }
        if (study["moyFinalGradient"]+std::abs(study["moyFinalGradient"].error)<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur la moyenne du gradient est respectée" << std::endl; farMin++;}
            iters[numeroPoint]+=study["iter"];
        }
        else
        {
            nonMin++;
            std::cout << "On n'est pas tombé sur un minimum" << std::endl;
        }

    }
    for(i=0;i<4;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i]; iters[i]/=proportions[i];}
        proportions[i]/=Sdouble(nbTirage);
    }
    std::cout << "La proportion pour (-2,1): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (-2,1): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,1): " << iters[0]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,-1): " << proportions[1] << std::endl;
    std::cout << "La distance moyenne à (2,-1): " << distances[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,-1): " << iters[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,-1): " << proportions[2] << std::endl;
    std::cout << "La distance moyenne à (0,-1): " << distances[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[2]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,1): " << proportions[3] << std::endl;
    std::cout << "La distance moyenne à (0,1): " << distances[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,1): " << iters[3]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << Sdouble(nonMin)/Sdouble(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << Sdouble(farMin)/Sdouble(nbTirage) << std::endl;

}
