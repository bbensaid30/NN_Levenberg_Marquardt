#ifndef TRAINING_ENTROPIE
#define TRAINING_ENTROPIE

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "propagation.h"
#include "scaling.h"
#include "utilities.h"

std::map<std::string,Sdouble> LM_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Sdouble& mu, Sdouble& factor, Sdouble const& eps, int const& maxIter, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LMF_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, Sdouble const& eps, int const& maxIter,
Sdouble const& RMin, Sdouble const& RMax, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LMUphill_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Sdouble const& eps, int const& maxIter, Sdouble const& RMin, Sdouble const& RMax, int const& b, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LMNielson_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Sdouble const& eps, int const& maxIter, Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& epsDiag,
bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> train_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& algo, Sdouble const& eps,
int const& maxIter, Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& epsDiag,
Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& sigma, bool const record=false, std::string const fileExtension="");

#endif
