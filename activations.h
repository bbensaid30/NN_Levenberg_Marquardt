#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <string>
#include <cmath>
#include <Eigen/Dense>

//Activations classiques
void linear(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);
void sigmoid(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);
void tanh(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);
void reLU(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);

//Activations pour les cas tests de la fonction de cout norme 2
//L'hypothèse est vérifiée pour tous les mins pour polyTwo et pour le min global pour polyThree
void polyTwo(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);
void polyThree(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);
void polyFour(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);
void expTwo(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);
void expFour(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);
void expFive(Eigen::MatrixXd& Z, Eigen::MatrixXd& S,double c=0);

//Activations pour les cas tests de la fonction de cout entropie
void ratTwo(Eigen::MatrixXd& Z, Eigen::MatrixXd& S, double c=0);
void cloche(Eigen::MatrixXd& Z, Eigen::MatrixXd& S);

void activation(std::string nameActivation, Eigen::MatrixXd& Z, Eigen::MatrixXd& S);

#endif

