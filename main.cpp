#include <iostream>
#include <string>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
    // Déclaration des variables du problème
    int const n0=3, nL=1, P=100;
    MatrixXd X = MatrixXd::Random(n0,P);
    MatrixXd Y = MatrixXd::Random(nL,P);

    int const L=2;
    int nbNeurons[L];
    string activations[L];
    MatrixXd weights[L];
    VectorXd bias[L];
    MatrixXd slopes[L];

    //Initialisation
    int i;
    nbNeurons[0]=2;
    weights[0]=MatrixXd::Random(nbNeurons[0],n0);
    bias[0]=VectorXd::Zero(nbNeurons[0]);
    activations[0]="sigmoid";
    for(i=1;i<L-1;i++)
    {
        nbNeurons[i]=2;
        weights[i]=MatrixXd::Random(nbNeurons[i],nbNeurons[i-1]);
        bias[i]=VectorXd::Zero(nbNeurons[i]);
        activations[i]="sigmoid";
    }
    nbNeurons[L-1]=nL;
    weights[L-1]=MatrixXd::Random(nbNeurons[L-1],nbNeurons[L-2]);
    bias[L-1]=VectorXd::Zero(nbNeurons[L-1]);
    activations[L-1]="sigmoid";

    return 0;

}
