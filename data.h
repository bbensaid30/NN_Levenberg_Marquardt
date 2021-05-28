#ifndef DATA
#define DATA

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <ctime>
#include <Eigen/Dense>

std::vector<Eigen::MatrixXd> sineWave(int const& nbPoints);
std::vector<Eigen::MatrixXd> squareWave(int const& nbPoints, double const frequence=1);
std::vector<Eigen::MatrixXd> sinc2(int const& nbPoints);
std::vector<Eigen::MatrixXd> exp2(int const& nbPoints);

std::vector<Eigen::MatrixXd> twoSpiral(int const& nbPoints, bool const noise=false);
std::vector<Eigen::MatrixXd> MNIST(std::string const& nameFileTrain, std::string const& nameFileTest);

std::vector<Eigen::MatrixXd> trainTestData(std::vector <Eigen::MatrixXd> const& data, double const percTrain = 0.9, bool const reproductible = true);

#endif
