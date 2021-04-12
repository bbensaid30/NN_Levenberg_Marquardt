#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <iostream>
#include <string>
#include <Eigen/Dense>

Eigen::MatrixXd sigmoid(Eigen::Ref<Eigen::MatrixXd> M);
Eigen::MatrixXd tanh(Eigen::Ref<Eigen::MatrixXd> M);
Eigen::MatrixXd reLu(Eigen::Ref<Eigen::MatrixXd> M);

#endif
