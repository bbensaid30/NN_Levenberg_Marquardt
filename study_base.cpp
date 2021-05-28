#include "study_base.h"

void denombrementMins(std::vector<Eigen::MatrixXd> const& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const generator, std::string algo, int const nbTirages, double const epsClose,
double const eps, int const maxIter, double mu, double const factor, double const RMin, double const RMax, int const b, double const alpha,
double const pas, double const Rlim, double const factorMin, double const power, double const alphaChap, double const epsDiag,
double const tau, double const beta, double const gamma, int const p, double const sigma, std::string const norm, double const radiusBall)
{
    int const N=globalIndices[2*L-1], PTrain=data[0].cols(), PTest=data[2].cols();
    unsigned seed;
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);
    std::vector<Eigen::MatrixXd> AsTest(L+1), slopesTest(L);
    AsTest[0]=data[2];
    Eigen::MatrixXd ETest(nbNeurons[L],PTest);

    std::map<std::string,double> study;

    Eigen::VectorXd point(N);
    std::vector<Eigen::VectorXd> points;
    std::vector<double> costs;
    std::vector<double> testingErrors;
    double testingError;
    bool newPoint;

    for(int i=0; i<nbTirages; i++)
    {
        std::cout << "Tirage: " << i << std::endl;
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,algo,eps,maxIter,mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tau,beta,gamma,
        p,sigma,norm,radiusBall);
        std::cout << "norme du gradient: " << study["finalGradient"] << std::endl;
        std::cout << "nb d'itérations: " << study["iter"] << std::endl;
        if(study["finalGradient"]<eps)
        {
            tabToVector(weights,bias,L,nbNeurons,globalIndices,point);
            newPoint = testPoint(point,points,epsClose);
            if(newPoint)
            {
                std::cout << "cout " << i << " : "<< study["finalCost"] << std::endl;
                costs.push_back(study["finalCost"]);
                fforward(data[2],data[3],L,PTest,nbNeurons,activations,weights,bias,AsTest,slopesTest,ETest);
                testingError = 0.5*ETest.squaredNorm();
                testingErrors.push_back(testingError);
                std::cout << "erreur prediction " << i << " : "<< testingError << std::endl;
                std::cout << "" << std::endl;
            }
        }
    }
    std::cout << "Il y a " << points.size() << " minimums " << std::endl;
}

void denombrementMins_entropie(std::vector<Eigen::MatrixXd> const& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const generator, std::string algo, int const nbTirages, double const epsClose,
double const eps, int const maxIter, double mu, double const factor, double const RMin, double const RMax, int const b, double const epsDiag, double const tau,
double const beta, double const gamma, int const p, double const sigma)
{
    int const N=globalIndices[2*L-1], PTrain=data[0].cols(), PTest=data[2].cols(), nL=nbNeurons[L];
    unsigned seed;
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);
    std::vector<Eigen::MatrixXd> AsTest(L+1), slopesTest(L);
    AsTest[0]=data[2];
    Eigen::MatrixXd ETest(nbNeurons[L],PTest);

    std::map<std::string,double> study;

    Eigen::VectorXd point(N);
    std::vector<Eigen::VectorXd> points;
    std::vector<double> costs;
    std::vector<double> testingErrors;
    double testingError;
    bool newPoint;

    for(int i=0; i<nbTirages; i++)
    {
        std::cout << "Tirage: " << i << std::endl;
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train_entropie(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,algo,eps,maxIter,mu,factor,RMin,RMax,b,epsDiag,tau,beta,gamma,p,sigma);
        std::cout << "norme du gradient: " << study["finalGradient"] << std::endl;
        std::cout << "nb d'itérations: " << study["iter"] << std::endl;
        if(study["finalGradient"]<eps)
        {
            tabToVector(weights,bias,L,nbNeurons,globalIndices,point);
            newPoint = testPoint(point,points,epsClose);
            if(newPoint)
            {
                std::cout << "cout " << i << " : "<< study["finalCost"] << std::endl;
                costs.push_back(study["finalCost"]);
                fforward(data[2],data[3],L,PTest,nbNeurons,activations,weights,bias,AsTest,slopesTest,ETest);
                testingError = entropie(data[3],AsTest[L],PTest,nL);
                testingErrors.push_back(testingError);
                std::cout << "erreur prediction " << i << " : "<< testingError << std::endl;
                std::cout << "" << std::endl;
            }
        }
    }
    std::cout << "Il y a " << points.size() << " minimums " << std::endl;
}

