#ifndef TESTLM
#define TESTLM

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "init.h"
#include "LMs.h"
#include "utilities.h"



void testLM_PolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& algo, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin,
Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight);

void testLM_PolyThree(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& algo, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin,
Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight);

void testLM_PolyFour(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& algo, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin,
Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight);

void testLM_Cloche(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& algo, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin,
Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight);

#endif
