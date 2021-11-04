#ifndef TESTSGD
#define TESTSGD

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "init.h"
#include "SGDs.h"
#include "utilities.h"



void testSGD_PolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& algo, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight);

#endif
