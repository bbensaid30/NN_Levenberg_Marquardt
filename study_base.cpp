#include "study_base.h"


int denombrementMinsInPlace(std::vector<Eigen::SMatrixXd> const& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const& generator, std::string const& algo, int const& nbTirages, Sdouble const& epsClose,
int const& nbDichotomie, Sdouble const& eps, int const& maxIter, Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag, Sdouble const& tau, Sdouble const& beta,
Sdouble const& gamma, int const& p, Sdouble const& sigma, std::string const& norm, Sdouble const& radiusBall, int const& tirageDisplay, int const& tirageMin, std::string const& strategy,
Sdouble const& flat)
{
    int PTrain=data[0].cols(), PTest=data[2].cols();
    unsigned seed;
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    std::vector<Eigen::SMatrixXd> AsTest(L+1), slopesTest(L);
    AsTest[0]=data[2];
    Eigen::SMatrixXd ETest(nbNeurons[L],PTest);

    std::map<std::string,Sdouble> study;

    std::vector<std::vector<Eigen::SMatrixXd>> weightsList;
    std::vector<std::vector<Eigen::SVectorXd>> biasList;
    std::vector<Sdouble> costs;
    std::vector<Sdouble> gradientNorms;
    std::vector<Sdouble> testingErrors;
    Sdouble testingError;

    Sdouble meanTraining, sdTraining, meanTest, sdTest;
    Sdouble costMinTraining, costMinTest, costMinBoth;
    int indiceMinTraining, indiceMinTest, indiceMinBoth;

    int const tirageMax=tirageMin+nbTirages;
    Sdouble costBoth;

    for(int i=tirageMin; i<tirageMax; i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,algo,eps,maxIter,mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tau,beta,gamma,
        p,sigma,norm,radiusBall);
        if(study["finalGradient"]+std::abs(study["finalGradient"].error)<eps)
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
        costBoth = (costs[i]+testingError)/2.0;
        if (costs[i]+std::abs(costs[i].error)<costMinTraining-std::abs(costMinTraining.error)){costMinTraining=costs[i]; indiceMinTraining=i;}
        if (testingError+std::abs(testingError.error)<costMinTest-std::abs(costMinTest.error)){costMinTest=testingError; indiceMinTest=i;}
        if (costBoth+std::abs(costBoth.error)<costMinBoth-std::abs(costMinBoth.error)){costMinBoth=costBoth; indiceMinBoth=i;}

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

    return weightsList.size();
}

int denombrementMinsPost(std::vector<Eigen::SMatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& algo, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eps, int const& tirageDisplay,
std::string const& strategy, Sdouble const& flat, std::string const folder, std::string const fileExtension)
{
    int const PTrain = data[0].cols();
    std::ostringstream epsStream;
    epsStream << eps;
    std::string epsString = epsStream.str();
    std::ostringstream PStream;
    PStream << PTrain;
    std::string PString = PStream.str();

    std::ifstream weightsMatrixesFlux(("Record/"+folder+"/weights_matrixes_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv").c_str());
    std::ifstream weightsVectorsFlux(("Record/"+folder+"/weights_vectors_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv").c_str());
    std::ifstream costFlux(("Record/"+folder+"/cost_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv").c_str());
    std::ifstream gradientNormFlux(("Record/"+folder+"/gradientNorm_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv").c_str());
    if(!weightsMatrixesFlux || !weightsVectorsFlux || !costFlux || !gradientNormFlux){std::cout << "Impossible d'ouvrir un des fichiers" << std::endl; exit(1);}

    int const PTest=data[2].cols();
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    for(int l=0; l<L; l++)
    {
        weights[l]=Eigen::SMatrixXd::Zero(nbNeurons[l+1],nbNeurons[l]);
        bias[l]=Eigen::SVectorXd::Zero(nbNeurons[l+1]);
    }
    std::vector<Eigen::SMatrixXd> AsTest(L+1), slopesTest(L);
    AsTest[0]=data[2];
    Eigen::SMatrixXd ETest(nbNeurons[L],PTest);

    std::map<std::string,Sdouble> study;

    std::vector<std::vector<Eigen::SMatrixXd>> weightsList;
    std::vector<std::vector<Eigen::SVectorXd>> biasList;
    Sdouble currentCost, currentGradientNorm;
    std::vector<Sdouble> costs;
    std::vector<Sdouble> gradientNorms;
    std::vector<Sdouble> testingErrors;
    Sdouble testingError;

    Sdouble meanTraining, sdTraining, meanTest, sdTest;
    Sdouble costMinTraining, costMinTest, costMinBoth;
    Sdouble costBoth;
    int indiceMinTraining, indiceMinTest, indiceMinBoth;

    int nbPoints = nbLines(costFlux);
    costFlux.clear(); costFlux.seekg(0,std::ios::beg);


    for(int i=0; i<nbPoints; i++)
    {
        for(int l=0; l<L; l++)
        {
            readMatrix(weightsMatrixesFlux,weights[l],nbNeurons[l+1],nbNeurons[l]);
            readVector(weightsMatrixesFlux,bias[l],nbNeurons[l+1]);
        }
        costFlux >> currentCost;
        gradientNormFlux >> currentGradientNorm;
        addPoint(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,data[0],data[1],L,PTrain,nbNeurons,globalIndices,activations,epsClose,
        nbDichotomie,flat,strategy);
        //if (i!=0 && i%tirageDisplay==0){std::cout << "Au bout de " << i << " points analysés, il y a " << weightsList.size() << " minimums" << std::endl; std::cout << " " << std::endl;}
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
        costBoth = (costs[i]+testingError)/2.0;
        if (costs[i]+std::abs(costs[i].error)<costMinTraining-std::abs(costMinTraining.error)){costMinTraining=costs[i]; indiceMinTraining=i;}
        if (testingError+std::abs(testingError.error)<costMinTest-std::abs(costMinTest.error)){costMinTest=testingError; indiceMinTest=i;}
        if (costBoth+std::abs(costBoth.error)<costMinBoth-std::abs(costMinBoth.error)){costMinBoth=costBoth; indiceMinBoth=i;}

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

    return weightsList.size();
}


int denombrementMins_entropie(std::vector<Eigen::SMatrixXd> const& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const& generator, std::string const& algo, int const& nbTirages, Sdouble const& epsClose,
int const& nbDichotomie, Sdouble const& eps, int const& maxIter, Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& epsDiag,
Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& sigma, int const& tirageDisplay, int const& tirageMin, std::string const& strategy,
Sdouble const& flat)
{
    int const nL=nbNeurons[L], PTrain=data[0].cols(), PTest=data[2].cols();
    unsigned seed;
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    std::vector<Eigen::SMatrixXd> AsTest(L+1), slopesTest(L);
    AsTest[0]=data[2];
    Eigen::SMatrixXd E_invTest(nbNeurons[L],PTest), E2_invTest(nbNeurons[L],PTest);

    std::map<std::string,Sdouble> study;

    std::vector<std::vector<Eigen::SMatrixXd>> weightsList;
    std::vector<std::vector<Eigen::SVectorXd>> biasList;
    std::vector<Sdouble> costs;
    std::vector<Sdouble> gradientNorms;
    std::vector<Sdouble> testingErrors;
    Sdouble testingError;

    Sdouble meanTraining, sdTraining, meanTest, sdTest;
    Sdouble costMinTraining=std::pow(10,3), costMinTest=std::pow(10,3), costMinBoth=std::pow(10,3);
    Sdouble costBoth;
    int indiceMinTraining, indiceMinTest, indiceMinBoth;

    int const tirageMax=tirageMin+nbTirages;

    for(int i=tirageMin; i<tirageMax; i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train_entropie(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,algo,eps,maxIter,mu,factor,RMin,RMax,b,epsDiag,tau,beta,gamma,p,sigma);
        if(study["finalGradient"]+std::abs(study["finalGradient"].error)<eps)
        {
            addPoint(weights,bias,weightsList,biasList,study["finalCost"],study["finalGradient"],costs,gradientNorms,data[0],data[1],L,PTrain,nbNeurons,globalIndices,activations,epsClose,
            nbDichotomie,flat,strategy);
        }
        if (i!=0 && i%tirageDisplay==0){std::cout << "Au bout de " << i << " tirages, il y a " << weightsList.size() << " minimums" << std::endl; std::cout << " " << std::endl;}
    }
    for (size_t i=0; i<weightsList.size(); i++)
    {
        costBoth = (costs[i]+testingError)/2.0;
        fforward_entropie(data[2],data[3],L,PTest,nbNeurons,activations,weightsList[i],biasList[i],AsTest,slopesTest,E_invTest,E2_invTest);
        testingError=entropie(data[3],AsTest[L],PTest,nL);
        testingErrors.push_back(testingError);
        costBoth = (costs[i]+testingError)/2.0;
        if (costs[i]+std::abs(costs[i].error)<costMinTraining-std::abs(costMinTraining.error)){costMinTraining=costs[i]; indiceMinTraining=i;}
        if (testingError+std::abs(testingError.error)<costMinTest-std::abs(costMinTest.error)){costMinTest=testingError; indiceMinTest=i;}
        if (costBoth+std::abs(costBoth.error)<costMinBoth-std::abs(costMinBoth.error)){costMinBoth=costBoth; indiceMinBoth=i;}

    }

    meanTraining=mean(costs); sdTraining=sd(costs,meanTraining);
    meanTest=mean(testingErrors); sdTest=sd(testingErrors,meanTest);

    std::cout << "Il y a " << weightsList.size() << " minimums " << std::endl;

    std::cout << "Moyenne coût d'entrainement: " << meanTraining << " +- " << sdTraining << std::endl;
    std::cout << "Le plus petit coût d'entraînement est de: " << costMinTraining << " de numéro " << indiceMinTraining << std::endl;

    std::cout << "Moyenne coût de test: " << meanTest << " +- " << sdTest << std::endl;
    std::cout << "Le plus petit coût de test est de: " << costMinTest << " de numéro " << indiceMinTest << std::endl;

    std::cout << "Le plus petit coût global est de: " << costMinBoth << " de numéro " << indiceMinBoth << std::endl;

    return weightsList.size();

}

void minsRecord(std::vector<Eigen::SMatrixXd> const& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<double> const& supParameters, std::string const& generator, std::string const& algo, int const& nbTirages,
Sdouble const& eps, int const& maxIter, Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& sigma, std::string const& norm, Sdouble const& radiusBall, int const& tirageMin,
int const& tirageDisplay, std::string const folder, std::string const fileExtension)
{

    int const PTrain = data[0].cols();
    std::ostringstream epsStream;
    epsStream << eps.number;
    std::string epsString = epsStream.str();
    std::ostringstream PStream;
    PStream << PTrain;
    std::string PString = PStream.str();

    std::ofstream weightsMatrixesFlux(("Record/"+folder+"/weights_matrixes_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv").c_str());
    std::ofstream weightsVectorsFlux(("Record/"+folder+"/weights_vectors_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv").c_str());
    std::ofstream costFlux(("Record/"+folder+"/cost_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv").c_str());
    std::ofstream gradientNormFlux(("Record/"+folder+"/gradientNorm_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv").c_str());
    if(!weightsMatrixesFlux || !weightsVectorsFlux || !costFlux || !gradientNormFlux){std::cout << "Impossible d'ouvrir un des fichiers" << std::endl; exit(1);}

    unsigned seed;
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    int const N = globalIndices[2*L-1];
    int jump;
    Eigen::SVectorXd point(N);

    std::map<std::string,Sdouble> study;

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
        if(study["finalGradient"]+std::abs(study["finalGradient"].error)<eps && !Sstd::isnan(study["finalCost"]) && !Sstd::isinf(study["finalCost"]))
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
            }

            costFlux << study["finalCost"].number << std::endl;
            costFlux << std::abs(study["finalCost"].error) << std::endl;
            costFlux << study["finalCost"].digits() << std::endl;

            gradientNormFlux << study["finalGradient"] << std::endl;
            gradientNormFlux << std::abs(study["finalGradient"].error) << std::endl;
            gradientNormFlux << study["finalGradient"].digits() << std::endl;

            weightsVectorsFlux << point.transpose() << std::endl;
            minAttain++;

        }
    }

}

