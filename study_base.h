#ifndef STUDY_BASE
#define STUDY_BASE

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <sstream>

#include "init.h"
#include "propagation.h"
#include "training.h"
#include "training_entropie.h"
#include "utilities.h"
#include "addStrategy.h"


void denombrementMinsInPlace(std::vector<Eigen::MatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const generator="uniform", std::string const algo="LM", int const nbTirages=10000,
double const epsClose=std::pow(10,-7), int const nbDichotomie = std::pow(2,4), double const eps=std::pow(10,-7), int const maxIter=2000, double mu=10, double const factor=10, double const RMin=0.25, double const RMax=0.75,
int const b=1, double const alpha=0.75, double const pas=0.1, double const Rlim=0.01, double const factorMin=std::pow(10,-3), double const power=2.0, double const alphaChap=2.0,
double const epsDiag= std::pow(10,-10), double const tau=0.1, double const beta=2, double const gamma=3, int const p=3, double const sigma=0.1, std::string const norm="2",
double const radiusBall=std::pow(10,0), int const tirageDisplay = 50, int const tirageMin=0, std::string const strategy="CostAll", double const flat=0.1);

void denombrementMinsPost(std::vector<Eigen::MatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const algo="LM", double const epsClose=std::pow(10,-7), int const nbDichotomie = std::pow(2,4), double const eps=std::pow(10,-7), int const tirageDisplay = 50,
int const tirageMin=0, std::string const strategy="CostAll", double const flat=0.1, std::string const fileExtension="");



void denombrementMins_entropie(std::vector<Eigen::MatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const generator="uniform", std::string algo="LM", int const nbTirages=10000,
double const epsClose=std::pow(10,-7), int const nbDichotomie = std::pow(2,4), double const eps=std::pow(10,-7), int const maxIter=2000, double mu=10, double const factor=10, double const RMin=0.25, double const RMax=0.75, int const b=1, double const epsDiag= std::pow(10,-10), double const tau=0.1,
double const beta=2, double const gamma=3, int const p=3, double const sigma=0.1, int const tirageDisplay = 50, int const tirageMin=0, std::string const strategy="CostAll", double const flat=0.1);



void minsRecord(std::vector<Eigen::MatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const generator="uniform", std::string algo="LM", int const nbTirages=10000,
double const eps=std::pow(10,-7), int const maxIter=2000, double mu=10, double const factor=10, double const RMin=0.25, double const RMax=0.75,
int const b=1, double const alpha=0.75, double const pas=0.1, double const Rlim=0.01, double const factorMin=std::pow(10,-3), double const power=2.0, double const alphaChap=2.0,
double const epsDiag= std::pow(10,-10), double const tau=0.1, double const beta=2, double const gamma=3, int const p=3, double const sigma=0.1, std::string const norm="2",
double const radiusBall=std::pow(10,0),int const tirageMin=0, int const tirageDisplay=100, std::string const fileExtension="");



#endif
