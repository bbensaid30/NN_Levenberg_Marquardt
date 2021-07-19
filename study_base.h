#ifndef STUDY_BASE
#define STUDY_BASE

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <cmath>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "init.h"
#include "propagation.h"
#include "training.h"
#include "training_entropie.h"
#include "utilities.h"
#include "addStrategy.h"


int denombrementMinsInPlace(std::vector<Eigen::SMatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const& generator, std::string const& algo, int const& nbTirages,
Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eps, int const& maxIter, Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax,
int const& b, Sdouble const& alpha, Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap,
Sdouble const& epsDiag, Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& sigma, std::string const& norm,
Sdouble const& radiusBall, int const& tirageDisplay, int const& tirageMin, std::string const& strategy, Sdouble const& flat);

int denombrementMinsPost(std::vector<Eigen::SMatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& algo, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eps, int const& tirageDisplay,
std::string const& strategy, Sdouble const& flat, std::string const folder="", std::string const fileExtension="");



int denombrementMins_entropie(std::vector<Eigen::SMatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const& generator, std::string const& algo, int const& nbTirages,
Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eps, int const& maxIter, Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b,
Sdouble const& epsDiag, Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& sigma, int const& tirageDisplay, int const& tirageMin,
std::string const& strategy, Sdouble const& flat);



void minsRecord(std::vector<Eigen::SMatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const& generator, std::string const& algo, int const& nbTirages,
Sdouble const& eps, int const& maxIter, Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax,
int const& b, Sdouble const& alpha, Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap,
Sdouble const& epsDiag, Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& sigma, std::string const& norm,
Sdouble const& radiusBall,int const& tirageMin, int const& tirageDisplay, std::string const folder="", std::string const fileExtension="");



#endif
