#ifndef PROPAGATION
#define PROPAGATION

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "activations.h"

//------------------------------------------------------------------Norme2----------------------------------------------------------------------------------------

void fforward(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E);

void backward(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E, Eigen::VectorXd& gradient, Eigen::MatrixXd& Q);

void backwardJacob(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& J);

//---------------------------------------------------------------Entropie---------------------------------------------------------------------------------------------

void fforward_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<Eigen::MatrixXd>& As,
std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E_inv, Eigen::MatrixXd& E2_inv);

void backward_entropie(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E_inv, Eigen::MatrixXd& E2_inv, Eigen::VectorXd& gradient, Eigen::MatrixXd& Q);

void backwardJacob_entropie(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E_inv, Eigen::MatrixXd& E2_inv, Eigen::MatrixXd& J, Eigen::MatrixXd& J2);

double entropie(Eigen::MatrixXd const& Y, Eigen::MatrixXd const& outputs, int const& P, int const& nL);

//--------------------------------------------------------------Générale---------------------------------------------------------------------------------------------

void solve(Eigen::VectorXd const& gradient, Eigen::MatrixXd const& hessian, Eigen::VectorXd& delta);

void update(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
Eigen::VectorXd const& delta);

#endif
