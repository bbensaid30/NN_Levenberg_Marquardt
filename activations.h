#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/Dense>

void sigmoid(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);
void tanh(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);
void reLU(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);

void activation(std::string nameActivation, Eigen::MatrixXd& Z, Eigen::MatrixXd& S);

#endif

