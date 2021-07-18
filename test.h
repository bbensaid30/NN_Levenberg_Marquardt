#ifndef TEST
#define TEST

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "init.h"
#include "training.h"
#include "training_entropie.h"
#include "utilities.h"



void testPolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& algo, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin,
Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight);

void testPolyThree(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& algo, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin,
Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight);

void testPolyFour(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& algo, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin,
Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight);

void testCloche(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& algo, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin,
Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight);

#endif
