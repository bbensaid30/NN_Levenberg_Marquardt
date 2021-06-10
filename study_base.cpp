#include "study_base.h"


void denombrementMins(std::vector<Eigen::MatrixXd> const& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const generator, std::string algo, int const nbTirages, double const epsClose, int const nbDichotomie,
double const eps, int const maxIter, double mu, double const factor, double const RMin, double const RMax, int const b, double const alpha,
double const pas, double const Rlim, double const factorMin, double const power, double const alphaChap, double const epsDiag,
double const tau, double const beta, double const gamma, int const p, double const sigma, std::string const norm, double const radiusBall, int const tirageDisplay, int const tirageMin,
std::string const strategy, double const eRelative)
{
    int PTrain=data[0].cols(), PTest=data[2].cols();
    unsigned seed;
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);
    std::vector<Eigen::MatrixXd> AsTest(L+1), slopesTest(L);
    AsTest[0]=data[2];
    Eigen::MatrixXd ETest(nbNeurons[L],PTest);

    std::map<std::string,double> study;

    std::vector<std::vector<Eigen::MatrixXd>> weightsList;
    std::vector<std::vector<Eigen::VectorXd>> biasList;
    std::vector<double> costs;
    std::vector<double> gradientNorms;
    std::vector<double> testingErrors;
    double testingError;

    double meanTraining, sdTraining, meanTest, sdTest;
    double costMinTraining, costMinTest, costMinBoth;
    int indiceMinTraining, indiceMinTest, indiceMinBoth;

    int const tirageMax=tirageMin+nbTirages;

    for(int i=tirageMin; i<tirageMax; i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,algo,eps,maxIter,mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tau,beta,gamma,
        p,sigma,norm,radiusBall);
        if(study["finalGradient"]<eps)
        {
            addPoint(weights,bias,weightsList,biasList,study["finalCost"],study["finalGradient"],costs,gradientNorms,data[0],data[1],L,PTrain,nbNeurons,globalIndices,activations,epsClose,
            nbDichotomie,eRelative,strategy);
        }
        if (i!=0 && i%tirageDisplay==0){std::cout << "Au bout de " << i << " tirages, il y a " << weightsList.size() << " minimums" << std::endl; std::cout << " " << std::endl;}
    }
    for (size_t i=0; i<weightsList.size(); i++)
    {
        fforward(data[2],data[3],L,PTest,nbNeurons,activations,weightsList[i],biasList[i],AsTest,slopesTest,ETest);
        testingError = 0.5*ETest.squaredNorm();
        testingErrors.push_back(testingError);
        if (i==0)
        {
            costMinTraining=costs[0]; indiceMinTraining=0;
            costMinTest=testingError; indiceMinTest=0;
            costMinBoth=(costMinTraining+testingError)/2.0; indiceMinBoth=0;
        }
        if (costs[i]<costMinTraining){costMinTraining=costs[i]; indiceMinTraining=i;}
        if (testingError<costMinTest){costMinTest=testingError; indiceMinTest=i;}
        if ((costs[i]+testingError)/2.0<costMinBoth){costMinBoth=(costs[i]+testingError)/2.0; indiceMinBoth=i;}

        std::cout << "cout d'entrainement: "  << i << ": " << costs[i] << std::endl;
        std::cout << "gradient : "  << i << ": " << gradientNorms[i] << std::endl;
    }

    meanTraining=mean(costs); sdTraining=sd(costs,meanTraining);
    meanTest=mean(testingErrors); sdTest=sd(testingErrors,meanTest);

    std::cout << "Il y a " << weightsList.size() << " minimums " << std::endl;

    std::cout << "Moyenne coût d'entrainement: " << meanTraining << " +- " << sdTraining << std::endl;
    std::cout << "Le plus petit coût d'entraînement est de: " << costMinTraining << " de numéro " << indiceMinTraining << std::endl;

    std::cout << "Moyenne coût de test: " << meanTest << " +- " << sdTest << std::endl;
    std::cout << "Le plus petit coût de test est de: " << costMinTest << " de numéro " << indiceMinTest << std::endl;

    std::cout << "Le plus petit coût global est de: " << costMinBoth << " de numéro " << indiceMinBoth << std::endl;
}

void denombrementMins_entropie(std::vector<Eigen::MatrixXd> const& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const generator, std::string algo, int const nbTirages, double const epsClose, int const nbDichotomie,
double const eps, int const maxIter, double mu, double const factor, double const RMin, double const RMax, int const b, double const epsDiag, double const tau,
double const beta, double const gamma, int const p, double const sigma, int const tirageDisplay, int const tirageMin, std::string const strategy, double const eRelative)
{
    int const nL=nbNeurons[L], PTrain=data[0].cols(), PTest=data[2].cols();
    unsigned seed;
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);
    std::vector<Eigen::MatrixXd> AsTest(L+1), slopesTest(L);
    AsTest[0]=data[2];
    Eigen::MatrixXd E_invTest(nbNeurons[L],PTest), E2_invTest(nbNeurons[L],PTest);

    std::map<std::string,double> study;

    std::vector<std::vector<Eigen::MatrixXd>> weightsList;
    std::vector<std::vector<Eigen::VectorXd>> biasList;
    std::vector<double> costs;
    std::vector<double> gradientNorms;
    std::vector<double> testingErrors;
    double testingError;

    double meanTraining, sdTraining, meanTest, sdTest;
    double costMinTraining=std::pow(10,3), costMinTest=std::pow(10,3), costMinBoth=std::pow(10,3);
    int indiceMinTraining, indiceMinTest, indiceMinBoth;

    int const tirageMax=tirageMin+nbTirages;

    for(int i=tirageMin; i<tirageMax; i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train_entropie(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,algo,eps,maxIter,mu,factor,RMin,RMax,b,epsDiag,tau,beta,gamma,p,sigma);
        if(study["finalGradient"]<eps)
        {
            addPoint(weights,bias,weightsList,biasList,study["finalCost"],study["finalGradient"],costs,gradientNorms,data[0],data[1],L,PTrain,nbNeurons,globalIndices,activations,epsClose,
            nbDichotomie,eRelative,strategy);
        }
        if (i!=0 && i%tirageDisplay==0){std::cout << "Au bout de " << i << " tirages, il y a " << weightsList.size() << " minimums" << std::endl; std::cout << " " << std::endl;}
    }
    for (size_t i=0; i<weightsList.size(); i++)
    {
        fforward_entropie(data[2],data[3],L,PTest,nbNeurons,activations,weightsList[i],biasList[i],AsTest,slopesTest,E_invTest,E2_invTest);
        testingError=entropie(data[3],AsTest[L],PTest,nL);
        testingErrors.push_back(testingError);
        if (costs[i]<costMinTraining){costMinTraining=costs[i]; indiceMinTraining=i;}
        if (testingError<costMinTest){costMinTest=testingError; indiceMinTest=i;}
        if ((costs[i]+testingError)/2.0<costMinBoth){costMinBoth=(costs[i]+testingError)/2.0; indiceMinBoth=i;}

    }

    meanTraining=mean(costs); sdTraining=sd(costs,meanTraining);
    meanTest=mean(testingErrors); sdTest=sd(testingErrors,meanTest);

    std::cout << "Il y a " << weightsList.size() << " minimums " << std::endl;

    std::cout << "Moyenne coût d'entrainement: " << meanTraining << " +- " << sdTraining << std::endl;
    std::cout << "Le plus petit coût d'entraînement est de: " << costMinTraining << " de numéro " << indiceMinTraining << std::endl;

    std::cout << "Moyenne coût de test: " << meanTest << " +- " << sdTest << std::endl;
    std::cout << "Le plus petit coût de test est de: " << costMinTest << " de numéro " << indiceMinTest << std::endl;

    std::cout << "Le plus petit coût global est de: " << costMinBoth << " de numéro " << indiceMinBoth << std::endl;

}

