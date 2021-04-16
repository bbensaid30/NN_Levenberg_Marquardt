#ifndef DATA
#define DATA

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <ctime>
#include <Eigen/Dense>

std::vector<Eigen::MatrixXd> sineWave(int nbPoints);

std::vector<Eigen::MatrixXd> trainTestData(std::vector <Eigen::MatrixXd> data, double percTrain = 0.9, bool reproductible = true);

#endif
