#ifndef SGDS
#define SGDS

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

#include <random>
#include <algorithm>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "propagation.h"
#include "utilities.h"
#include "eigenExtension.h"
#include "perte.h"

std::map<std::string,Sdouble> SGD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, int const& batch_size, Sdouble const& eps, int const& maxIter, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> Momentum(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& eps, int const& maxIter, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> AdaGrad(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& eps, int const& maxIter, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> RMSProp(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta2, Sdouble const& eps, int const& maxIter, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> Adam(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> AMSGrad(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> train_SGD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& learning_rate, int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter, bool const record=false, std::string const fileExtension="");



#endif
