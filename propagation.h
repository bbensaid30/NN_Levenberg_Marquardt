#ifndef PROPAGATION
#define PROPAGATION

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "activations.h"
#include "perte.h"


//------------------------------------------------------------------ Propagation directe ----------------------------------------------------------------------------------------

void fforward(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes);

Sdouble risk(Eigen::SMatrixXd const& Y, int const& P, Eigen::SMatrixXd const& output_network, std::string const& type_perte);

//-------------------------------------------------------------------- Rétropropagation -----------------------------------------------------------------------------------------------

void backward(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SVectorXd& gradient, std::string const& type_perte);

void QSO_backward(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes,Eigen::SVectorXd& gradient, Eigen::SMatrixXd& Q, std::string const& type_perte);

void QSO_backwardJacob(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& J);


//-------------------------------------------------------------- Mise à jour ---------------------------------------------------------------------------------------------

void solve(Eigen::SVectorXd const& gradient, Eigen::SMatrixXd const& hessian, Eigen::SVectorXd& delta, std::string const method = "HouseholderQR");

void update(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Eigen::SVectorXd const& delta);

#endif
