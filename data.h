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
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"
#include <EigenRand/EigenRand>

std::vector<Eigen::SMatrixXd> sineWave(int const& nbPoints);
std::vector<Eigen::SMatrixXd> squareWave(int const& nbPoints, Sdouble const frequence=1);
std::vector<Eigen::SMatrixXd> sinc1(int const& nbPoints);

std::vector<Eigen::SMatrixXd> sinc2(int const& nbPoints);
std::vector<Eigen::SMatrixXd> exp2(int const& nbPoints);

std::vector<Eigen::SMatrixXd> twoSpiral(int const& nbPoints);
std::vector<Eigen::SMatrixXd> MNIST(std::string const& nameFileTrain, std::string const& nameFileTest);

std::vector<Eigen::SMatrixXd> trainTestData(std::vector<Eigen::SMatrixXd> const& data, Sdouble const& percTrain = 0.9, bool const reproductible = true);

#endif
