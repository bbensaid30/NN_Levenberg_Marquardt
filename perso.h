#ifndef PERSO
#define PERSO

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
#include "eigenExtension.h"

std::map<std::string,Sdouble> EulerRichardson(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> Momentum_ER(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate_init, Sdouble const& beta1_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LM_ER(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& mu_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> train_Perso(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& learning_rate_init, Sdouble const& beta1_init, Sdouble const& mu_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");


#endif
