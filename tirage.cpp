#include "tirage.h"

std::vector<std::map<std::string,Sdouble>> tirages(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
bool const tracking, bool const track_continuous)
{
    int const PTest = data[2].cols();

    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    std::vector<Eigen::SMatrixXd> AsTest(L+1);
    AsTest[0]=data[2];
    std::vector<Eigen::SMatrixXd> slopes(L);

    Sdouble costTest;

    int const tirageMax=tirageMin+nbTirages;

    int i;
    std::vector<std::map<std::string,Sdouble>> studies(nbTirages);

    for(i=0;i<nbTirages;i++)
    {

        initialisation(nbNeurons,weights,bias,supParameters,generator,i);

        studies[i] = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxIter,
        learning_rate,clip,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
        costTest = risk(data[3],PTest,AsTest[L],type_perte);
        studies[i]["cost_test"] = costTest;
        studies[i]["num_tirage"] = i;

        std::cout << "On est au tirage: " << i << std::endl;
    }

    return studies;

}

void minsRecord(std::vector<std::map<std::string,Sdouble>> studies, std::string const& folder, std::string const& fileEnd, Sdouble const& eps)
{

    std::ofstream costFlux(("Record/"+folder+"/cost_"+fileEnd).c_str());
    std::ofstream costTestFlux(("Record/"+folder+"/costTest_"+fileEnd).c_str());

    if(!costFlux || !costTestFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    int const nbTirages = studies.size();
    int nonConv=0, div=0;

    std::map<std::string,Sdouble> study;

    for(int i=0; i<nbTirages; i++)
    {
        study = studies[i];

        if(study["finalGradient"]+std::abs(study["finalGradient"].error)<eps && !Sstd::isnan(study["finalGradient"]) && !Sstd::isinf(study["finalGradient"]) && !numericalNoise(study["finalGradient"]))
        {

            costFlux << study["num_tirage"] << std::endl;
            costFlux << study["iter"].number << std::endl;
            costFlux << study["finalCost"].number << std::endl;
            costFlux << std::abs(study["finalCost"].error) << std::endl;

            costTestFlux << study["num_tirage"] << std::endl;
            costTestFlux << study["iter"].number << std::endl;
            costTestFlux << study["cost_test"].number << std::endl;
            costTestFlux << std::abs(study["cost_test"].error) << std::endl;
        }
        else
        {
            if(Sstd::abs(study["finalGradient"])>1 || Sstd::isnan(study["finalGradient"]))
            {
                div++;
            }
            else
            {
                nonConv++;
            }
        }

    }

    std::cout << "Proportion de divergence: " << Sdouble(div)/Sdouble(nbTirages) << std::endl;
    std::cout << "Proportion de non convergence: " << Sdouble(nonConv)/Sdouble(nbTirages) << std::endl;

}

void predictionsRecord(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
std::string const& folder, std::string const fileExtension, bool const tracking, bool const track_continuous)
{

    int const PTrain = data[0].cols();
    int const PTest = data[2].cols();

    std::string const fileEnd = informationFile(PTrain,PTest,L,nbNeurons,activations,type_perte,algo,supParameters,generator,tirageMin,nbTirages,eps,maxIter);

    std::ofstream costFlux(("Record/"+folder+"/cost_"+fileEnd).c_str());
    std::ofstream costTestFlux(("Record/"+folder+"/costTest_"+fileEnd).c_str());

    std::ofstream inputsFlux(("Record/"+folder+"/inputs_"+fileEnd).c_str());
    std::ofstream bestTrainFlux(("Record/"+folder+"/bestTrain_"+fileEnd).c_str());
    std::ofstream bestTestFlux(("Record/"+folder+"/bestTest_"+fileEnd).c_str());
    std::ofstream moyTrainFlux(("Record/"+folder+"/moyTrain_"+fileEnd).c_str());
    std::ofstream moyTestFlux(("Record/"+folder+"/moyTest_"+fileEnd).c_str());

    std::ofstream trackingFlux(("Record/"+folder+"/tracking_"+fileEnd).c_str());
    std::ofstream trackContinuousFlux(("Record/"+folder+"/track_continuous_"+fileEnd).c_str());


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
    int nonConv=0, div=0;

    int const tirageMax=tirageMin+nbTirages;
    int minAttain=0, nMin;

    for(int i=tirageMin; i<tirageMax; i++)
    {
        if(i!=0 && i%100==0)
        {
            std::cout << "On est au tirage" << i << std::endl;
        }

        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxIter,
        learning_rate,clip,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);
        if(study["finalGradient"]+std::abs(study["finalGradient"].error)<eps && !Sstd::isnan(study["finalCost"]) && !Sstd::isinf(study["finalCost"]) && !numericalNoise(study["finalGradient"]))
        {
            fforward(L,PTrain,nbNeurons,activations,weights,bias,AsTrain,slopes);

            costFlux << i << std::endl;
            costFlux << study["iter"].number << std::endl;
            costFlux << study["finalCost"].number << std::endl;
            costFlux << std::abs(study["finalCost"].error) << std::endl;

            fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
            costTest = risk(data[3],PTest,AsTest[L],type_perte);

            costTestFlux << i << std::endl;
            costTestFlux << study["iter"].number << std::endl;
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

            minAttain++; nMin=0;

        }
        else
        {
            if(Sstd::abs(study["finalGradient"])>1 || Sstd::isnan(study["finalGradient"]))
            {
                div++; nMin=-3;
            }
            else
            {
                nonConv++; nMin=-2;
            }
        }

        if(tracking)
        {
            trackingFlux << nMin << std::endl;
            trackingFlux << study["iter"].number << std::endl;
            trackingFlux << study["prop_entropie"].number << std::endl;
        }

        if(track_continuous)
        {
            trackContinuousFlux << nMin << std::endl;
            trackContinuousFlux << study["iter"].number << std::endl;
            trackContinuousFlux << study["continuous_entropie"].number << std::endl;
        }
    }

    std::cout << "Proportion de divergence: " << Sdouble(div)/Sdouble(nbTirages) << std::endl;
    std::cout << "Proportion de non convergence: " << Sdouble(nonConv)/Sdouble(nbTirages) << std::endl;

    bestTrainFlux << bestPredictionsTrain << std::endl;
    bestTestFlux << bestPredictionsTest << std::endl;
    moyTrainFlux << moyPredictionsTrain/Sdouble(minAttain) << std::endl;
    moyTestFlux << moyPredictionsTest/Sdouble(minAttain) << std::endl;

}


std::string informationFile(int const& PTrain, int const& PTest, int const& L, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter, std::string const fileExtension)
{
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
    std::ostringstream maxIterStream;
    maxIterStream << maxIter;
    std::string maxIterString = maxIterStream.str();

    std::string archi = "";
    for(int l=0; l<L; l++)
    {
        archi += std::to_string(nbNeurons[l+1]);
        archi+="("; archi += activations[l]; archi += ")";
        archi+="-";
    }

    int tailleParameters = supParameters.size();
    std::string gen = generator; gen+="(";
    if(tailleParameters>0)
    {
        for(int s=0; s<tailleParameters; s++)
        {
            gen += std::to_string(supParameters[s]); gen+=",";
        }
    }
    gen+=")";


    return algo+"("+fileExtension+")"+archi+"(eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+", maxIter="+maxIterString+")"+ gen +".csv";
}
