#include "study.h"

void minsRecord(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
std::string const folder, std::string const fileExtension)
{

    int const PTrain = data[0].cols();
    int const PTest = data[2].cols();
    std::ostringstream epsStream;
    epsStream << eps.number;
    std::string epsString = epsStream.str();
    std::ostringstream PTrainStream;
    PTrainStream << PTrain;
    std::string PTrainString = PTrainStream.str();
    std::ostringstream PTestStream;
    PTestStream << PTest;
    std::string PTestString = PTestStream.str();
    std::ostringstream tirageMinStream;
    tirageMinStream << tirageMin;
    std::string tirageMinString = tirageMinStream.str();
    std::ostringstream nbTiragesStream;
    nbTiragesStream << nbTirages;
    std::string nbTiragesString = nbTiragesStream.str();

    std::string const fileEnd = algo+"_"+fileExtension+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv";
    std::ofstream costFlux(("Record/"+folder+"/cost_"+fileEnd).c_str());
    std::ofstream costTestFlux(("Record/"+folder+"/costTest_"+fileEnd).c_str());

    if(!costFlux || !costTestFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    unsigned seed;
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    int const N = globalIndices[2*L-1];

    std::vector<Eigen::SMatrixXd> AsTest(L+1); AsTest[0]=data[2];
    std::vector<Eigen::SMatrixXd> slopes(L);

    Sdouble costTest;

    std::map<std::string,Sdouble> study;

    int const tirageMax=tirageMin+nbTirages;

    for(int i=tirageMin; i<tirageMax; i++)
    {
        if(i!=0 && i%100==0)
        {
            std::cout << "On est au tirage" << i << std::endl;
        }

        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,
        famille_algo,algo,eps,maxIter,
        learning_rate,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag);

        if(study["finalGradient"]+std::abs(study["finalGradient"].error)<eps && !Sstd::isnan(study["finalCost"]) && !Sstd::isinf(study["finalCost"]))
        {

            costFlux << i << std::endl;
            costFlux << study["finalCost"].number << std::endl;
            costFlux << std::abs(study["finalCost"].error) << std::endl;

            fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
            costTest = risk(data[3],PTest,AsTest[L],type_perte);

            costTestFlux << i << std::endl;
            costTestFlux << costTest.number << std::endl;
            costTestFlux << std::abs(costTest.error) << std::endl;

        }
    }

}

void predictionsRecord(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
std::string const folder, std::string const fileExtension)
{

    int const PTrain = data[0].cols();
    int const PTest = data[2].cols();
    std::ostringstream epsStream;
    epsStream << eps.number;
    std::string epsString = epsStream.str();
    std::ostringstream PTrainStream;
    PTrainStream << PTrain;
    std::string PTrainString = PTrainStream.str();
    std::ostringstream PTestStream;
    PTestStream << PTest;
    std::string PTestString = PTestStream.str();
    std::ostringstream tirageMinStream;
    tirageMinStream << tirageMin;
    std::string tirageMinString = tirageMinStream.str();
    std::ostringstream nbTiragesStream;
    nbTiragesStream << nbTirages;
    std::string nbTiragesString = nbTiragesStream.str();

    std::string const fileEnd = algo+"_"+fileExtension+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+").csv";
    std::ofstream costFlux(("Record/"+folder+"/cost_"+fileEnd).c_str());
    std::ofstream costTestFlux(("Record/"+folder+"/costTest_"+fileEnd).c_str());

    std::ofstream inputsFlux(("Record/"+folder+"/inputs_"+fileEnd).c_str());
    std::ofstream bestTrainFlux(("Record/"+folder+"/bestTrain_"+fileEnd).c_str());
    std::ofstream bestTestFlux(("Record/"+folder+"/bestTest_"+fileEnd).c_str());
    std::ofstream moyTrainFlux(("Record/"+folder+"/moyTrain_"+fileEnd).c_str());
    std::ofstream moyTestFlux(("Record/"+folder+"/moyTest_"+fileEnd).c_str());

    if(!costFlux || !costTestFlux || !inputsFlux || !bestTrainFlux || !bestTestFlux || !moyTrainFlux || !moyTestFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    unsigned seed;
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    int const N = globalIndices[2*L-1];

    std::vector<Eigen::SMatrixXd> AsTrain(L+1); AsTrain[0]=data[0];
    std::vector<Eigen::SMatrixXd> AsTest(L+1); AsTest[0]=data[2];
    std::vector<Eigen::SMatrixXd> slopes(L);

    Eigen::SMatrixXd bestPredictionsTrain(nbNeurons[L],PTrain), bestPredictionsTest(nbNeurons[L],PTest);
    Eigen::SMatrixXd moyPredictionsTrain(nbNeurons[L],PTrain), moyPredictionsTest(nbNeurons[L],PTest);
    moyPredictionsTrain.setZero(); moyPredictionsTest.setZero();

    Sdouble costMin=10000, costTest;

    std::map<std::string,Sdouble> study;

    int const tirageMax=tirageMin+nbTirages;
    int minAttain=0;

    for(int i=tirageMin; i<tirageMax; i++)
    {
        if(i!=0 && i%100==0)
        {
            std::cout << "On est au tirage" << i << std::endl;
        }

        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxIter,
        learning_rate,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag);
        if(study["finalGradient"]+std::abs(study["finalGradient"].error)<eps && !Sstd::isnan(study["finalCost"]) && !Sstd::isinf(study["finalCost"]))
        {
            fforward(L,PTrain,nbNeurons,activations,weights,bias,AsTrain,slopes);

            costFlux << i << std::endl;
            costFlux << study["finalCost"].number << std::endl;
            costFlux << std::abs(study["finalCost"].error) << std::endl;

            fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
            costTest = risk(data[3],PTest,AsTest[L],type_perte);

            costTestFlux << i << std::endl;
            costTestFlux << costTest.number << std::endl;
            costTestFlux << std::abs(costTest.error) << std::endl;

            if(study["finalCost"] < costMin)
            {
                costMin = study["finalCost"];

                bestPredictionsTrain = AsTrain[L];
                bestPredictionsTest = AsTest[L];
            }
            moyPredictionsTrain += AsTrain[L];
            moyPredictionsTest += AsTest[L];

            minAttain++;

        }
    }

    bestTrainFlux << bestPredictionsTrain << std::endl;
    bestTestFlux << bestPredictionsTest << std::endl;
    moyTrainFlux << moyPredictionsTrain/Sdouble(minAttain) << std::endl;
    moyTestFlux << moyPredictionsTest/Sdouble(minAttain) << std::endl;

}
