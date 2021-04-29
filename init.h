#ifndef INIT
#define INIT

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <Eigen/Dense>

void simple(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias);
void uniform(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const& a, double const& b);
void normal(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const& mu, double const& sigma);

void initialisation(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<double> const& supParameters,
std::string generator="simple");

#endif
