#ifndef TEST
#define TEST

#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "init.h"
#include "training.h"
#include "training_entropie.h"
#include "utilities.h"


void testPolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int nbTirage=10000, std::string const algo="LMUphill", double mu=10, double factor=10,
double const Rlim=std::pow(10,-4), double const RMin=0.25, double const RMax=0.75, double const epsDiag=std::pow(10,-8), int const b=2, double const factorMin=std::pow(10,-8),
double const power=2.0, double const alphaChap=3.0, double const eps=std::pow(10,-7), int maxIter=20000, double epsNeight=std::pow(10,-7));

void testPolyThree(std::string const& distribution, std::vector<double> const& supParameters, int nbTirage=10000, std::string const algo="LMUphill", double mu=10, double factor=10,
double const Rlim=std::pow(10,-4), double const RMin=0.25, double const RMax=0.75, double const epsDiag=std::pow(10,-8), int const b=2, double const factorMin=std::pow(10,-8),
double const power=2.0, double const alphaChap=3.0, double const eps=std::pow(10,-7), int maxIter=20000, double epsNeight=std::pow(10,-7));

void testCloche(std::string const& distribution, std::vector<double> const& supParameters, int nbTirage=10000, std::string const algo="LMUphill", double mu=10, double factor=10,
double const Rlim=std::pow(10,-4), double const RMin=0.25, double const RMax=0.75, double const epsDiag=std::pow(10,-8), int const b=2, double const factorMin=std::pow(10,-8),
double const power=2.0, double const alphaChap=3.0, double const eps=std::pow(10,-7), int maxIter=20000, double epsNeight=std::pow(10,-7));

#endif
