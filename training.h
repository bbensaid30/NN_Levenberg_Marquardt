#ifndef TRAINING
#define TRAINING

#include <string>
#include <Eigen/Dense>
#include "activations.h"

void propagation(Eigen::MatrixXd& A, Eigen::MatrixXd const& Y, int const L, int const nbNeurons[], int const P, std::string const activations[], Eigen::MatrixXd const weights[],
Eigen::VectorXd const bias[], Eigen::MatrixXd As[], Eigen::MatrixXd slopes[], Eigen::MatrixXd E, Eigen::VectorXd dz, Eigen::VectorXd Jpm, Eigen::VectorXd gradient, Eigen::MatrixXd Q);

void train();

#endif
