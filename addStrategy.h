#ifndef ADDSTRATEGY
#define ADDSTRATEGY

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <map>

#include "propagation.h"
#include "utilities.h"

bool addSimple(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
double const epsClose=std::pow(10,-3));

bool addCostCloser(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie=std::pow(2,4), double const eRelative=0.1);

bool addCostAll(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie=std::pow(2,4), double const eRelative=0.1);

bool addCostAllSet(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie=std::pow(2,4), double const eRelative=0.1);

bool addCostSd(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie=std::pow(2,4), double const eRelative=0.1);

bool addCostSdAbs(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie=std::pow(2,4), double const eAbs=0.1);

bool addCostMedian(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie=std::pow(2,4), double const eRelative=0.1);



bool addCostSdNonEquivalent(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie=std::pow(2,4), double const eRelative=0.1);




bool addPoint(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie=std::pow(2,4), double const flat=0.1, std::string const strategy="CostAll");


#endif
