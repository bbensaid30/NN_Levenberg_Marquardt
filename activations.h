#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <string>
#include <cmath>
#include <Eigen/Dense>

void linear(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);
void sigmoid(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);
void tanh(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);
void reLU(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);



void polyTwo(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);
void polyThree(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);
void polyFour(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);
void expTwo(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);
void expFour(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);
void expFive(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);


void activation(std::string nameActivation, Eigen::MatrixXd& Z, Eigen::MatrixXd& S);

#endif

