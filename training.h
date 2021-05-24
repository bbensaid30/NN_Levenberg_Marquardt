#ifndef TRAINING
#define TRAINING

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "propagation.h"
#include "utilities.h"
#include "scaling.h"

std::map<std::string,double> LM(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double mu=10, double const factor=10, double const eps=std::pow(10,-7), int const maxIter=2000,bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMBall(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double mu=10, double factor=10, double const eps=std::pow(10,-7),
int const maxIter=2000, std::string const norm="2", double const radiusBall=std::pow(10,0), bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMF(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps=std::pow(10,-7), int const maxIter=2000, double const RMin=0.25, double const RMax=0.75, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMMore(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps=std::pow(10,-7), int const maxIter=2000, double const sigma=0.1, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMNielson(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps=std::pow(10,-7), int const maxIter=2000, double const tau=0.1, double const beta=2, double const gamma=3, int const p=3, double const epsDiag=std::pow(10,-3),
bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMUphill(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps=std::pow(10,-7), int const maxIter=2000, double const RMin=0.25, double const RMax=0.75, int const b=1, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMPerso(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const eps=std::pow(10,-7), int const maxIter=2000,
double const RMin=0.25, double const RMax=0.75, int const b=1, double const epsDiag= std::pow(10,-10), bool const record=false, std::string const fileExtension="");

//------------------------------------------------------------- Méthodes plus coûteuses utilisant explicitement la jacobienne -----------------------------------------------------------------------------------------------

std::map<std::string,double> LMGeodesic(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps=std::pow(10,-7), int const maxIter=2000, double const RMin=0.25, double const RMax=0.75, int const b=1, double const alpha=0.75, double const pas=0.1,
bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMJynian(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps=std::pow(10,-7), int const maxIter=2000, double const Rlim=0.01, double const RMin=0.25, double const RMax=0.75, double const factorMin=std::pow(10,-3), double const power=2.0, double const alphaChap=2.0,
bool const record=false, std::string const fileExtension="");

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

std::map<std::string,double> train(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const algo,
double const eps=std::pow(10,-7), int const maxIter=2000, double mu=10, double const factor=10, double const RMin=0.25, double const RMax=0.75, int const b=1, double const alpha=0.75,
double const pas=0.1, double const Rlim=0.01, double const factorMin=std::pow(10,-3), double const power=2.0, double const alphaChap=2.0, double const epsDiag= std::pow(10,-10),
double const tau=0.1, double const beta=2, double const gamma=3, int const p=3, double const sigma=0.1, std::string const norm="2", double const radiusBall=std::pow(10,0),
bool const record=false, std::string const fileExtension="");


#endif
