#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "activations.h"

int main()
{
    // Déclaration des variables du problème
    int const n0=3, nL=1, P=100;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n0,P);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(nL,P);

    int const L=2;
    int nbNeurons[L];
    std::string activations[L];
    Eigen::MatrixXd weights[L];
    Eigen::MatrixXd bias[L];
    Eigen::MatrixXd slopes[L];

    //Initialisation
    int i;
    nbNeurons[0]=2;
    weights[0]=Eigen::MatrixXd::Random(nbNeurons[0],n0);
    bias[0]=Eigen::MatrixXd::Zero(nbNeurons[0],1);
    activations[0]="sigmoid";
    for(i=1;i<L-1;i++)
    {
        nbNeurons[i]=2;
        weights[i]=Eigen::MatrixXd::Random(nbNeurons[i],nbNeurons[i-1]);
        bias[i]=Eigen::MatrixXd::Zero(nbNeurons[i],1);
        activations[i]="sigmoid";
    }
    nbNeurons[L-1]=nL;
    weights[L-1]=Eigen::MatrixXd::Random(nbNeurons[L-1],nbNeurons[L-2]);
    bias[L-1]=Eigen::MatrixXd::Zero(nbNeurons[L-1],1);
    activations[L-1]="sigmoid";



    return 0;

}
