#ifndef STUDY
#define STUDY

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "init.h"
#include "propagation.h"
#include "training.h"


void minsRecord(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
std::string const folder, std::string const fileExtension);

void predictionsRecord(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
std::string const folder, std::string const fileExtension);

#endif
