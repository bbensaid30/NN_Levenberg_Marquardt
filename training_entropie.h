#ifndef TRAINING_ENTROPIE
#define TRAINING_ENTROPIE

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "propagation.h"

std::map<std::string,double> LM_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double mu=10, double factor=10, double const eps=std::pow(10,-7), int const maxIter=2000,bool record=false, std::string const fileExtension="");

#endif
