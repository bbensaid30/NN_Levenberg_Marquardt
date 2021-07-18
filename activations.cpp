#include "activations.h"

void linear(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    S = Eigen::SMatrixXd::Constant(Z.rows(),Z.cols(),1);
}

void sigmoid(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    Z = (1+(-Z).array().exp()).inverse();
    S = Z.array()*(1-Z.array());
}

void softmax(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    Z = Z.array().exp();
    Eigen::SVectorXd sumExp = Z.colwise().sum().cwiseInverse();

    Z = Z * sumExp.asDiagonal();
    S = Z.array()*(1-Z.array());

}

void tanh(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    Z = Z.array().tanh();
    S = 1-Z.array().pow(2);
}

void reLU(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    Eigen::SMatrixXd U = Eigen::SMatrixXd::Constant(Z.rows(),Z.cols(),1);
    Z = Z.cwiseMax(0);
    S = (Z.array()>0).select(U,0);
}


void activation(std::string nameActivation, Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    if (nameActivation == "sigmoid")
    {
        sigmoid(Z,S);
    }
    else if (nameActivation == "linear")
    {
        linear(Z,S);
    }
    else if(nameActivation == "tanh")
    {
        tanh(Z,S);
    }
    else if(nameActivation == "reLU")
    {
        reLU(Z,S);
    }
    else if(nameActivation == "softmax")
    {
        softmax(Z,S);
    }

    else if(nameActivation == "polyTwo")
    {
        polyTwo(Z,S);
    }
    else if(nameActivation=="polyThree")
    {
        polyThree(Z,S);
    }
    else if(nameActivation=="polyFour")
    {
        polyFour(Z,S);
    }
    else if(nameActivation=="expTwo")
    {
        expTwo(Z,S);
    }
    else if(nameActivation=="expFour")
    {
        expFour(Z,S);
    }
    else if(nameActivation=="expFive")
    {
        expFive(Z,S);
    }
    else if(nameActivation=="ratTwo")
    {
        ratTwo(Z,S,1);
    }
    else if(nameActivation=="cloche")
    {
        cloche(Z,S);
    }
    else
    {
        linear(Z,S);
    }
}


void polyTwo(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    S = 2*Sstd::exp(c/2)*Z.array();
    Z = Sstd::exp(c/2)*(Z.array().pow(2)-1);
}

void polyThree(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    S = 6*Sstd::exp(c/2)*Z.array()*(Z.array()-1);
    Z = Sstd::exp(c/2)*(2*Z.array().pow(3)-3*Z.array().pow(2)+5);
}

void polyFour(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    S = 4*Sstd::exp(c/2)*Z.array()*(Z.array().pow(2)-1);
    Z = Sstd::exp(c/2)*(Z.array().pow(4)-2*Z.array().pow(2)+3);
}

void expTwo(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    Eigen::SMatrixXd inter = (1/2*(Z.array().pow(2)-2*Z.array().pow(2)+c)).exp();
    S = (Z.array()-1)*inter.array();
    Z = inter;
}

void expFour(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    Eigen::SMatrixXd inter = (1/2*(Z.array().pow(4)-2*Z.array().pow(2)+c)).exp();
    S = 2*Z.array()*(Z.array().pow(2)-1)*inter.array();
    Z = inter;
}

void expFive(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    Eigen::SMatrixXd inter = (1/2*(6*Z.array().pow(5)-15*Z.array().pow(4)-10*Z.array().pow(3)+30*Z.array().pow(2)+c)).exp();
    S = 15*Z.array()*(Z.array().pow(3)-2*Z.array().pow(2)-Z.array()+2)*inter.array();
    Z = inter;
}

void ratTwo(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    Sdouble const normalization = 2.0/(c+2+Sstd::sqrt(c*c+4));
    Eigen::SMatrixXd poly = Z.array().pow(2)-2*Z.array()+2;
    S = (-2*Z.array().pow(2)+2*(2-c)*Z.array()+2*c)*(poly.array().pow(2)).inverse(); S*=normalization;
    Z = (Z.array().pow(2)+c)*poly.array().inverse(); Z*=normalization;
}

void cloche(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    Eigen::SMatrixXd exp2 = (-0.5*Z.array().pow(2)).exp();
    S = -Z.array()*exp2.array();
    Z = exp2;
}
