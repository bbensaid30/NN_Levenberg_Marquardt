#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "training.h"

int main()
{
    // Déclaration des variables du problème
    int const n0=3, nL=1, P=100;
    int N=0;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n0,P);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(nL,P);

    int const L=2;
    int nbNeurons[L+1];
    std::string activations[L];
    Eigen::MatrixXd weights[L];
    Eigen::VectorXd bias[L];
    Eigen::MatrixXd As[L+1];
    Eigen::MatrixXd slopes[L];

    //Initialisation
    int l;
    nbNeurons[0]=n0;
    weights[0]=Eigen::MatrixXd::Random(nbNeurons[1],nbNeurons[0]);
    bias[0]=Eigen::VectorXd::Zero(nbNeurons[1]);
    activations[0]="sigmoid";
    N+=(nbNeurons[0]+1)*nbNeurons[1];
    for(l=1;l<L-1;l++)
    {
        nbNeurons[l]=2;
        weights[l]=Eigen::MatrixXd::Random(nbNeurons[l+1],nbNeurons[l]);
        bias[l]=Eigen::VectorXd::Zero(nbNeurons[l+1]);
        activations[l]="sigmoid";
        N+=(nbNeurons[l]+1)*nbNeurons[l+1];
    }
    nbNeurons[L]=nL;
    weights[L-1]=Eigen::MatrixXd::Random(nbNeurons[L],nbNeurons[L-1]);
    bias[L-1]=Eigen::VectorXd::Zero(nbNeurons[L]);
    activations[L-1]="sigmoid";
    N+=(nbNeurons[L-1]+1)*nbNeurons[L];

    Eigen::MatrixXd A = X;
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd dz = Eigen::VectorXd::Zero(nbNeurons[L]);
    Eigen::VectorXd Jpm(N);
    Eigen::VectorXd gradient(N);
    Eigen::MatrixXd Q(N,N);

    propagation(A,Y,L,nbNeurons,P,activations,weights,bias,As,slopes,E,dz,Jpm,gradient,Q);

    std::cout << As[1] << std::endl;

    return 0;

}
