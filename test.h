#ifndef TEST
#define TEST

#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "init.h"
#include "training.h"
#include "utilities.h"


void testPolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int nbTirage=10000, double mu=10, double factor=10,
double const eps=std::pow(10,-7), int maxIter=2000, double epsNeight=std::pow(10,-7));

void testPolyThree(std::string const& distribution, std::vector<double> const& supParameters, int nbTirage=10000, double mu=10, double factor=10,
double const eps=std::pow(10,-7), int maxIter=2000, double epsNeight=std::pow(10,-7));

#endif
