#include "activations.h"

void sigmoid(Eigen::MatrixXd& Z, Eigen::MatrixXd& S)
{
    Z = (1+(-Z).array().exp()).inverse();
    S = Z.array()*(1-Z.array());
}

void tanh(Eigen::MatrixXd& Z, Eigen::MatrixXd& S)
{
    Z = Z.array().tanh();
    S = 1-Z.array().pow(2);
}

//Pas encore vectorisée
void reLU(Eigen::MatrixXd& Z, Eigen::MatrixXd& S)
{
    Eigen::MatrixXd U = Eigen::MatrixXd::Constant(Z.rows(),Z.cols(),1);
    Z = Z.cwiseMax(0);
    S = (Z.array()>0).select(U,0);
}

void activation(std::string nameActivation, Eigen::MatrixXd& Z, Eigen::MatrixXd& S)
{
    if (nameActivation == "sigmoid")
    {
        sigmoid(Z,S);
    }
    else if(nameActivation == "tanh")
    {
        tanh(Z,S);
    }
    else if(nameActivation == "reLU")
    {
        reLU(Z,S);
    }
    else
    {
        sigmoid(Z,S);
    }
}








