#include <iostream>
#include <string>
#include <iterator>
#include <Eigen/Dense>
#include "training.h"

int main()
{
    // Problem variables
    int const n0=3, nL=1, P=100;
    int N=0;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n0,P);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(nL,P);
    int const L=2;
    int nbNeurons[L+1];
    int globalIndices[2*L];
    std::string activations[L];
    Eigen::MatrixXd weights[L];
    Eigen::VectorXd bias[L];

    //Initialisation
    int l;
    nbNeurons[0]=n0; nbNeurons[1]=2;
    weights[0]=Eigen::MatrixXd::Random(nbNeurons[1],nbNeurons[0]);
    bias[0]=Eigen::VectorXd::Zero(nbNeurons[1]);
    activations[0]="sigmoid";
    N+=nbNeurons[0]*nbNeurons[1]; globalIndices[0]=N; N+=nbNeurons[1]; globalIndices[1]=N;
    for(l=1;l<L-1;l++)
    {
        nbNeurons[l+1]=2;
        weights[l]=Eigen::MatrixXd::Random(nbNeurons[l+1],nbNeurons[l]);
        bias[l]=Eigen::VectorXd::Zero(nbNeurons[l+1]);
        activations[l]="sigmoid";
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }
    nbNeurons[L]=nL;
    weights[L-1]=Eigen::MatrixXd::Random(nbNeurons[L],nbNeurons[L-1]);
    bias[L-1]=Eigen::VectorXd::Zero(nbNeurons[L]);
    activations[L-1]="sigmoid";
    N+=nbNeurons[L-1]*nbNeurons[L]; globalIndices[2*L-2]=N; N+=nbNeurons[L]; globalIndices[2*L-1]=N;

    //Training
    double mu=10, factor=10, eps=std::pow(10,-3);
    int maxIter=2000;
    train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter);

    return 0;

}
