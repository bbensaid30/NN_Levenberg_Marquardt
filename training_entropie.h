#ifndef TRAINING_ENTROPIE
#define TRAINING_ENTROPIE

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "propagation.h"
#include "scaling.h"
#include "utilities.h"

std::map<std::string,double> LM_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double mu=10, double factor=10, double const eps=std::pow(10,-7), int const maxIter=2000,bool record=false, std::string const fileExtension="");

std::map<std::string,double> LMF_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const eps=std::pow(10,-7), int const maxIter=20000,
double const RMin=0.25, double const RMax=0.75, bool const record=false, std::string fileExtension="");

std::map<std::string,double> LMUphill_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps=std::pow(10,-7), int const maxIter=2000, double const RMin=0.25, double const RMax=0.75, int const b=1, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMNielson_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps=std::pow(10,-7), int const maxIter=2000, double const tau=0.1, double const beta=2, double const gamma=3, int const p=3, double const epsDiag=std::pow(10,-3),
bool const record=false, std::string const fileExtension="");

std::map<std::string,double> train_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string algo="LM", double const eps=std::pow(10,-7), int const maxIter=2000,
double mu=10, double const factor=10, double const RMin=0.25, double const RMax=0.75, int const b=1, double const epsDiag= std::pow(10,-10), double const tau=0.1,
double const beta=2, double const gamma=3, int const p=3, double const sigma=0.1, bool const record=false, std::string const fileExtension="");

#endif
