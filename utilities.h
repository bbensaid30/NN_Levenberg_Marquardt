#ifndef UTILITIES
#define UTILITIES

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <numeric>



int proportion(Eigen::VectorXd const& currentPoint, std::vector<Eigen::VectorXd> const& points, std::vector<double>& proportions,  std::vector<double>& distances, double const& epsNeight);

double mean(std::vector<double> const& values);
double sd(std::vector<double> const& values, double const& moy);
double median(std::vector<double>& values);

double distance(std::vector<Eigen::MatrixXd> const& weightsPrec, std::vector<Eigen::VectorXd> const& biasPrec,
std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::string const norm="2");

double cosVector(Eigen::VectorXd const& v1, Eigen::VectorXd const& v2);

void convexCombination(std::vector<Eigen::MatrixXd> const& weights1, std::vector<Eigen::VectorXd> const& bias1, std::vector<Eigen::MatrixXd> const& weights2,
std::vector<Eigen::VectorXd> const& bias2, std::vector<Eigen::MatrixXd>& weightsInter, std::vector<Eigen::VectorXd>& biasInter, int const& L, double const lambda=0.5);

void tabToVector(std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd> const& bias, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
Eigen::VectorXd& point);

void standardization(Eigen::MatrixXd& X);

int nbLines(std::ifstream& flux);
void readMatrix(std::ifstream& flux, Eigen::MatrixXd& result, int const& nbRows, int const& nbCols);
void readVector(std::ifstream& flux, Eigen::VectorXd& result, int const& nbRows);


#endif
