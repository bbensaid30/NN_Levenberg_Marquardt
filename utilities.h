#ifndef UTILITIES
#define UTILITIES

#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>


int proportion(Eigen::VectorXd const& currentPoint, std::vector<Eigen::VectorXd> const& points, std::vector<double>& proportions,  std::vector<double>& distances, double const& epsNeight);

double distance(std::vector<Eigen::MatrixXd> const& weightsPrec, std::vector<Eigen::VectorXd> const& biasPrec,
std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::string norm="2");

double cosVector(Eigen::VectorXd const& v1, Eigen::VectorXd const& v2);


void tabToVector(std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd> const& bias, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
Eigen::VectorXd& point);
bool testPoint(Eigen::VectorXd const& point, std::vector<Eigen::VectorXd>& points, double const epsClose=std::pow(10,-7));

#endif
