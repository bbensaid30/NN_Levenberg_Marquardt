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


//------------------------------------------------------------------Norme2----------------------------------------------------------------------------------------

void fforward(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& E);

void backward(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& E, Eigen::SVectorXd& gradient, Eigen::SMatrixXd& Q);

void backwardJacob(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& J);

//---------------------------------------------------------------Entropie---------------------------------------------------------------------------------------------

void fforward_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::vector<Eigen::SMatrixXd>& As,
std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& E_inv, Eigen::SMatrixXd& E2_inv);

void backward_entropie(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& E_inv, Eigen::SMatrixXd& E2_inv, Eigen::SVectorXd& gradient, Eigen::SMatrixXd& Q);

void backwardJacob_entropie(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& E_inv, Eigen::SMatrixXd& E2_inv, Eigen::SMatrixXd& J, Eigen::SMatrixXd& J2);

Sdouble entropie(Eigen::SMatrixXd const& Y, Eigen::SMatrixXd const& outputs, int const& P, int const& nL);

//--------------------------------------------------------------Générale---------------------------------------------------------------------------------------------

void solve(Eigen::SVectorXd const& gradient, Eigen::SMatrixXd const& hessian, Eigen::SVectorXd& delta, std::string const method = "HouseholderQR");

void update(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Eigen::SVectorXd const& delta);

#endif
