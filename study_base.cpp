#include "study_base.h"


void denombrementMinsInPlace(std::vector<Eigen::MatrixXd> const& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const generator, std::string const algo, int const nbTirages, double const epsClose, int const nbDichotomie,
double const eps, int const maxIter, double mu, double const factor, double const RMin, double const RMax, int const b, double const alpha,
double const pas, double const Rlim, double const factorMin, double const power, double const alphaChap, double const epsDiag,
double const tau, double const beta, double const gamma, int const p, double const sigma, std::string const norm, double const radiusBall, int const tirageDisplay, int const tirageMin,
std::string const strategy, double const flat)
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
            nbDichotomie,flat,strategy);
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

void denombrementMinsPost(std::vector<Eigen::MatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const algo, double const epsClose, int const nbDichotomie, double const eps, int const tirageDisplay,
int const tirageMin, std::string const strategy, double const flat, std::string const fileExtension)
{
    std::ostringstream epsStream;
    epsStream << eps;
    std::string epsString = epsStream.str();
    std::ifstream weightsMatrixesFlux(("Record/weights_matrixes_"+algo+"_"+fileExtension+"(eps="+epsString+").csv").c_str());
    std::ifstream costFlux(("Record/cost_"+algo+"_"+fileExtension+"(eps="+epsString+").csv").c_str());
    std::ifstream gradientNormFlux(("Record/gradientNorm_"+algo+"_"+fileExtension+"(eps="+epsString+").csv").c_str());
    if(!weightsMatrixesFlux || !costFlux || !gradientNormFlux){std::cout << "Impossible d'ouvrir un des fichiers" << std::endl; exit(1);}

    int const PTrain = data[0].cols(), PTest=data[2].cols();
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);
    for(int l=0; l<L; l++)
    {
        weights[l]=Eigen::MatrixXd::Zero(nbNeurons[l+1],nbNeurons[l]);
        bias[l]=Eigen::VectorXd::Zero(nbNeurons[l+1]);
    }
    std::vector<Eigen::MatrixXd> AsTest(L+1), slopesTest(L);
    AsTest[0]=data[2];
    Eigen::MatrixXd ETest(nbNeurons[L],PTest);

    std::map<std::string,double> study;

    std::vector<std::vector<Eigen::MatrixXd>> weightsList;
    std::vector<std::vector<Eigen::VectorXd>> biasList;
    double currentCost, currentGradientNorm;
    std::vector<double> costs;
    std::vector<double> gradientNorms;
    std::vector<double> testingErrors;
    double testingError;

    double meanTraining, sdTraining, meanTest, sdTest;
    double costMinTraining, costMinTest, costMinBoth;
    int indiceMinTraining, indiceMinTest, indiceMinBoth;

    int nbPoints = nbLines(costFlux);
    costFlux.clear(); costFlux.seekg(0,std::ios::beg);


    for(int i=0; i<nbPoints; i++)
    {   std::cout << i << std::endl;
        for(int l=0; l<L; l++)
        {
            readMatrix(weightsMatrixesFlux,weights[l],nbNeurons[l+1],nbNeurons[l]);
            readVector(weightsMatrixesFlux,bias[l],nbNeurons[l+1]);
        }
        costFlux >> currentCost;
        gradientNormFlux >> currentGradientNorm;
        addPoint(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,data[0],data[1],L,PTrain,nbNeurons,globalIndices,activations,epsClose,
        nbDichotomie,flat,strategy);
        if (i!=0 && i%tirageDisplay==0){std::cout << "Au bout de " << i << " points analysés, il y a " << weightsList.size() << " minimums" << std::endl; std::cout << " " << std::endl;}
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
double const beta, double const gamma, int const p, double const sigma, int const tirageDisplay, int const tirageMin, std::string const strategy, double const flat)
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
            nbDichotomie,flat,strategy);
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

void minsRecord(std::vector<Eigen::MatrixXd> const& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const generator, std::string algo, int const nbTirages,
double const eps, int const maxIter, double mu, double const factor, double const RMin, double const RMax, int const b, double const alpha,
double const pas, double const Rlim, double const factorMin, double const power, double const alphaChap, double const epsDiag,
double const tau, double const beta, double const gamma, int const p, double const sigma, std::string const norm, double const radiusBall, int const tirageMin, int const tirageDisplay,
std::string const fileExtension)
{
    std::ostringstream epsStream;
    epsStream << eps;
    std::string epsString = epsStream.str();

    std::ofstream weightsMatrixesFlux(("Record/weights_matrixes_"+algo+"_"+fileExtension+"(eps="+epsString+").csv").c_str());
    std::ofstream weightsVectorsFlux(("Record/weights_vectors_"+algo+"_"+fileExtension+"(eps="+epsString+").csv").c_str());
    std::ofstream costFlux(("Record/cost_"+algo+"_"+fileExtension+"(eps="+epsString+").csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_"+algo+"_"+fileExtension+"(eps="+epsString+").csv").c_str());
    if(!weightsMatrixesFlux || !weightsVectorsFlux || !costFlux || !gradientNormFlux){std::cout << "Impossible d'ouvrir un des fichiers" << std::endl; exit(1);}

    unsigned seed;
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);
    int const N = globalIndices[2*L-1];
    int jump;
    Eigen::VectorXd point(N);

    std::map<std::string,double> study;

    int const tirageMax=tirageMin+nbTirages;
    int minAttain=0;

    for(int i=tirageMin; i<tirageMax; i++)
    {
        if(i!=0 && i%tirageDisplay==0)
        {
            std::cout << "Au bout de " << i << " tirages " << minAttain << " mins atteints " << std::endl;
        }

        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,algo,eps,maxIter,mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tau,beta,gamma,
        p,sigma,norm,radiusBall);
        if(study["finalGradient"]<eps && !std::isnan(study["finalCost"]))
        {
            for(int l=0; l<L; l++)
            {
                weightsMatrixesFlux << weights[l] << std::endl;
                weightsMatrixesFlux << bias[l] << std::endl;

                jump=nbNeurons[l+1]*nbNeurons[l];
                weights[l].resize(jump,1);
                point.segment(globalIndices[2*l]-jump,jump)=weights[l];
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                jump=nbNeurons[l+1];
                point.segment(globalIndices[2*l+1]-jump,jump)=bias[l];

                costFlux << study["finalCost"] << std::endl;
                gradientNormFlux << study["finalGradient"] << std::endl;
            }
            weightsVectorsFlux << point.transpose() << std::endl;
            minAttain++;

        }
    }

}

