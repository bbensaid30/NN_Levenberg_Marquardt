#ifndef TRAINING_ENTROPIE
#define TRAINING_ENTROPIE

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "activations.h"

void fforward_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<Eigen::MatrixXd>& As,
std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E_inv, Eigen::MatrixXd& E2_inv);

void backward_entropie(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E_inv, Eigen::MatrixXd& E2_inv, Eigen::VectorXd& gradient, Eigen::MatrixXd& Q);

void update_entropie(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
Eigen::VectorXd const& gradient, Eigen::MatrixXd const& hessian);

double entropie(Eigen::MatrixXd const& Y, Eigen::MatrixXd const& outputs, int const& P, int const& nL);

std::map<std::string,double> train_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double mu=10, double factor=10, double const eps=std::pow(10,-7), int const maxIter=2000,bool record=false, std::string const fileExtension="");

#endif
