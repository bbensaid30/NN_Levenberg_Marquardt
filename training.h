#ifndef TRAINING
#define TRAINING

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "activations.h"

void fforward(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, int const nbNeurons[], std::string const activations[],
Eigen::MatrixXd weights[], Eigen::VectorXd bias[], Eigen::MatrixXd As[], Eigen::MatrixXd slopes[], Eigen::MatrixXd& E);

void backward(int const& L, int const& P, int const nbNeurons[], int const globalIndices[], Eigen::MatrixXd weights[], Eigen::VectorXd bias[],
Eigen::MatrixXd As[], Eigen::MatrixXd slopes[], Eigen::MatrixXd& E, Eigen::VectorXd& gradient, Eigen::MatrixXd& Q);

void update(int const& L, int const nbNeurons[], int const globalIndices[], Eigen::MatrixXd weights[], Eigen::VectorXd bias[],
Eigen::VectorXd const& gradient, Eigen::MatrixXd const& hessian);

void train(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const nbNeurons[], int const globalIndices[], std::string const activations[],
Eigen::MatrixXd weights[], Eigen::VectorXd bias[], double& mu, double& factor, double const& eps, int const& maxIter);

#endif
