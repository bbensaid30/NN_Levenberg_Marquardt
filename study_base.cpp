#include "study_base.h"

void denombrementMins(std::vector<Eigen::MatrixXd> const& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const generator, std::string algo, int const nbTirages, double const epsClose,
double const eps, int const maxIter, double mu, double const factor, double const RMin, double const RMax, int const b, double const alpha,
double const pas, double const Rlim, double const factorMin, double const power, double const alphaChap, double const epsDiag,
double const tau, double const beta, double const gamma, int const p, double const sigma, std::string const norm, double const radiusBall)
{
    int const N=globalIndices[L], P=data[0].cols();
    unsigned seed;
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);
    std::vector<Eigen::MatrixXd> As(L+1), slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);

    std::map<std::string,double> study;

    Eigen::VectorXd point(N);
    std::vector<Eigen::VectorXd> points;
    std::vector<double> costs;
    std::vector<double> testingErrors;
    bool newPoint;

    for(int i=0; i<nbTirages; i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,algo,eps,maxIter,mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tau,beta,gamma,
        p,sigma,norm,radiusBall);
        if(study["finalGradient"]<eps)
        {
            tabToVector(weights,bias,L,nbNeurons,globalIndices,point);
            newPoint = testPoint(point,points,epsClose);
            if(newPoint)
            {
                costs.push_back(study["finalCost"]);
                fforward(data[2],data[3],L,P,nbNeurons,activations,weights,bias,As,slopes,E);
                testingErrors.push_back(0.5*E.squaredNorm());
            }
        }
    }

    std::cout << "Il y a " << points.size() << " minimums " << std::endl;
}
