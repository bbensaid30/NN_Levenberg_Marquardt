#include "test.h"

void testPolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int nbTirage, double mu, double factor, double const eps, int maxIter, double epsNeight)
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

    int i;
    std::map<std::string,double> study;
    Eigen::VectorXd currentPoint(2);
    std::vector<Eigen::VectorXd> points(4);
    std::vector<double> proportions(4,0.0), distances(4,0.0), iters(4,0.0), backs(4,0.0);
    int numeroPoint;

    points[0]=Eigen::VectorXd::Zero(2); points[1]=Eigen::VectorXd::Zero(2); points[2]=Eigen::VectorXd::Zero(2); points[3]=Eigen::VectorXd::Zero(2);
    points[0](0)=-2; points[0](1)=1; points[1](0)=2; points[1](1)=-1; points[2](0)=0; points[2](1)=-1; points[3](0)=0; points[3](1)=1;
    for(i=0;i<nbTirage;i++)
    {
        initialisation(nbNeurons,weights,bias,supParameters,distribution);
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter);

        currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
        numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
        if(numeroPoint<0){std::cout << "On n'est pas tombé sur un minimum" << std::endl;}
        iters[numeroPoint]+=study["iter"];
        backs[numeroPoint]+=study["propBack"];
    }
    for(i=0;i<4;i++)
    {
        distances[i]/=proportions[i];
        proportions[i]/=(double)nbTirage;
        iters[i]/=(double)nbTirage;
        backs[i]/=(double)nbTirage;
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

}

void testPolyThree(std::string const& distribution, std::vector<double> const& supParameters, int nbTirage, double mu, double factor, double const eps, int maxIter, double epsNeight)
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

    int i;
    std::map<std::string,double> study;
    Eigen::VectorXd currentPoint(2);
    std::vector<Eigen::VectorXd> points(5);
    std::vector<double> proportions(5,0.0), distances(5,0.0), iters(5,0.0), backs(5,0.0);
    int numeroPoint;

    points[0]=Eigen::VectorXd::Zero(2); points[1]=Eigen::VectorXd::Zero(2); points[2]=Eigen::VectorXd::Zero(2); points[3]=Eigen::VectorXd::Zero(2); points[4]=Eigen::VectorXd::Zero(2);
    points[0](0)=0; points[0](1)=-1; points[1](0)=2; points[1](1)=-1; points[2](0)=-2; points[2](1)=1; points[3](0)=-1; points[3](1)=0; points[4](0)=0; points[4](1)=1;
    for(i=0;i<nbTirage;i++)
    {
        initialisation(nbNeurons,weights,bias,supParameters,distribution);
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter);

        currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
        numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
        if(numeroPoint<0){std::cout << "On n'est pas tombé sur un minimum" << std::endl;}
        iters[numeroPoint]+=study["iter"];
        backs[numeroPoint]+=study["propBack"];
    }
    for(i=0;i<5;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i];}
        proportions[i]/=(double)nbTirage;
        iters[i]/=(double)nbTirage;
        backs[i]/=(double)nbTirage;
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

}

