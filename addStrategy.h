#ifndef ADDSTRATEGY
#define ADDSTRATEGY

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "propagation.h"
#include "utilities.h"

bool addSimple(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Sdouble const& epsClose);

bool addCostCloser(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eRelative);

bool addCostAll(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eRelative);

bool addCostAllSet(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eRelative);

bool addCostSd(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eRelative);

bool addCostSdAbs(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eAbs);

bool addCostMedian(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eRelative);






bool addPoint(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& flat, std::string const& strategy);


#endif
