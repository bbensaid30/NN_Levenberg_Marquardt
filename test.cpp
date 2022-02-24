#include "test.h"

void test_PolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor, Sdouble const& Rlim, Sdouble const& RMin,
Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas,
Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{

    std::ofstream gradientNormFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"iter"+".csv").c_str());
    std::ofstream initFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream trackingFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"tracking"+".csv").c_str());
    std::ofstream trackContinuousFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"track_continuous"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux || !trackingFlux || !trackContinuousFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::SMatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    std::vector<Eigen::SMatrixXd> weightsInit(L);
    std::vector<Eigen::SVectorXd> biasInit(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyTwo";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    int i; unsigned seed;
    std::map<std::string,Sdouble> study;
    Eigen::SVectorXd currentPoint(2);
    std::vector<Eigen::SVectorXd> points(4);
    std::vector<Sdouble> proportions(4,0.0), distances(4,0.0), iters(4,0.0);
    int numeroPoint, farMin=0, nonMin=0;

    points[0]=Eigen::SVectorXd::Zero(2); points[1]=Eigen::SVectorXd::Zero(2); points[2]=Eigen::SVectorXd::Zero(2); points[3]=Eigen::SVectorXd::Zero(2);
    points[0](0)=-2; points[0](1)=1; points[1](0)=2; points[1](1)=-1; points[2](0)=0; points[2](1)=-1; points[3](0)=0; points[3](1)=1;

    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxIter,
        learning_rate,clip,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        if (study["finalGradient"]+std::abs(study["finalGradient"].error)<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée" << std::endl; farMin++;}
            else{iters[numeroPoint]+=study["iter"];}
        }
        else
        {
            nonMin++;
            std::cout << "On n'est pas tombé sur un minimum" << std::endl;

            if(Sstd::abs(study["finalGradient"])>1 || Sstd::isnan(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence ou précision: " << i << std::endl;
                numeroPoint = -2;
            }
        }

        if(record)
        {
            gradientNormFlux << numeroPoint << std::endl;
            gradientNormFlux << study["finalGradient"].number << std::endl;
            gradientNormFlux << study["finalGradient"].error << std::endl;

            iterFlux << numeroPoint << std::endl;
            iterFlux << study["iter"].number << std::endl;

            initFlux << numeroPoint << std::endl;
            initFlux << weightsInit[0](0,0).number << std::endl;
            initFlux << biasInit[0](0).number << std::endl;
        }

        if(tracking)
        {
            trackingFlux << numeroPoint << std::endl;
            trackingFlux << study["iter"].number << std::endl;
            trackingFlux << study["prop_entropie"].number << std::endl;
            trackingFlux << study["prop_initial_ineq"].number << std::endl;
        }

        if(track_continuous)
        {
            trackContinuousFlux << numeroPoint << std::endl;
            trackContinuousFlux << study["iter"].number << std::endl;
            trackContinuousFlux << study["continuous_entropie"].number << std::endl;
        }
    }

    for(i=0;i<4;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i]; iters[i]/=proportions[i];}
        proportions[i]/=Sdouble(nbTirage);
    }

    std::cout << "La proportion pour (-2,1): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (-2,1): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,1): " << iters[0]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,-1): " << proportions[1] << std::endl;
    std::cout << "La distance moyenne à (2,-1): " << distances[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,-1): " << iters[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,-1): " << proportions[2] << std::endl;
    std::cout << "La distance moyenne à (0,-1): " << distances[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[2]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,1): " << proportions[3] << std::endl;
    std::cout << "La distance moyenne à (0,1): " << distances[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,1): " << iters[3]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << Sdouble(nonMin)/Sdouble(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << Sdouble(farMin)/Sdouble(nbTirage) << std::endl;

}

void test_PolyThree(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor, Sdouble const& Rlim, Sdouble const& RMin,
Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas,
Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{
    std::ofstream gradientNormFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"iter"+".csv").c_str());
    std::ofstream initFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream trackingFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"tracking"+".csv").c_str());
    std::ofstream trackContinuousFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"track_continuous"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux || !trackingFlux || !trackContinuousFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::SMatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    std::vector<Eigen::SMatrixXd> weightsInit(L);
    std::vector<Eigen::SVectorXd> biasInit(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyThree";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    int i; unsigned seed;
    std::map<std::string,Sdouble> study;
    Eigen::SVectorXd currentPoint(2);
    std::vector<Eigen::SVectorXd> points(4);
    std::vector<Sdouble> proportions(4,0.0), distances(4,0.0), iters(4,0.0);
    int numeroPoint, farMin=0, nonMin=0;

    points[0]=Eigen::SVectorXd::Zero(2); points[1]=Eigen::SVectorXd::Zero(2); points[2]=Eigen::SVectorXd::Zero(2); points[3]=Eigen::SVectorXd::Zero(2);
    points[0](0)=-2; points[0](1)=1; points[1](0)=2; points[1](1)=-1; points[2](0)=0; points[2](1)=-1; points[3](0)=0; points[3](1)=1;
    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxIter,learning_rate,clip,seuil,beta1,beta2,batch_size,mu,factor,RMin,RMax,
        b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking, track_continuous);

        if (study["finalGradient"]+std::abs(study["finalGradient"].error)<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint<0){farMin++;}
            else{iters[numeroPoint]+=study["iter"];}
        }

        else
        {
            nonMin++;
            std::cout << "On n'est pas tombé sur un minimum" << i << std::endl;

            if(Sstd::abs(study["finalGradient"])>1 || Sstd::isnan(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence ou précision: " << i << std::endl;
                numeroPoint = -2;
            }
        }

        if(record)
        {
            gradientNormFlux << numeroPoint << std::endl;
            gradientNormFlux << study["finalGradient"].number << std::endl;
            gradientNormFlux << study["finalGradient"].error << std::endl;

            iterFlux << numeroPoint << std::endl;
            iterFlux << study["iter"].number << std::endl;

            initFlux << numeroPoint << std::endl;
            initFlux << weightsInit[0](0,0).number << std::endl;
            initFlux << biasInit[0](0).number << std::endl;
        }

        if(tracking)
        {
            trackingFlux << numeroPoint << std::endl;
            trackingFlux << study["iter"].number << std::endl;
            trackingFlux << study["prop_entropie"].number << std::endl;
            trackingFlux << study["prop_initial_ineq"].number << std::endl;
        }

        if(track_continuous)
        {
            trackContinuousFlux << numeroPoint << std::endl;
            trackContinuousFlux << study["iter"].number << std::endl;
            trackContinuousFlux << study["continuous_entropie"].number << std::endl;
        }
    }

    for(i=0;i<4;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i]; iters[i]/=proportions[i];}
        proportions[i]/=Sdouble(nbTirage);
    }

    std::cout << "La proportion pour (-2,1): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (-2,1): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,1): " << iters[0]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,-1): " << proportions[1] << std::endl;
    std::cout << "La distance moyenne à (2,-1): " << distances[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,-1): " << iters[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,-1): " << proportions[2] << std::endl;
    std::cout << "La distance moyenne à (0,-1): " << distances[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[2]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,1): " << proportions[3] << std::endl;
    std::cout << "La distance moyenne à (0,1): " << distances[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,1): " << iters[3]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << Sdouble(nonMin)/Sdouble(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << Sdouble(farMin)/Sdouble(nbTirage) << std::endl;

}


void test_PolyFour(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor, Sdouble const& Rlim, Sdouble const& RMin,
Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas,
Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{

    std::ofstream gradientNormFlux(("Record/polyFour/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/polyFour/"+setHyperparameters+"/"+algo+"_"+"iter"+".csv").c_str());
    std::ofstream initFlux(("Record/polyFour/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream trackingFlux(("Record/polyFour/"+setHyperparameters+"/"+algo+"_"+"tracking"+".csv").c_str());
    std::ofstream trackContinuousFlux(("Record/polyFour/"+setHyperparameters+"/"+algo+"_"+"track_continuous"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux || !trackingFlux || !trackContinuousFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::SMatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    std::vector<Eigen::SMatrixXd> weightsInit(L);
    std::vector<Eigen::SVectorXd> biasInit(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyFour";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    int i; unsigned seed;
    std::map<std::string,Sdouble> study;
    Eigen::SVectorXd currentPoint(2);
    std::vector<Eigen::SVectorXd> points(4);
    std::vector<Sdouble> proportions(4,0.0), distances(4,0.0), iters(4,0.0);
    int numeroPoint, farMin=0, nonMin=0;

    points[0]=Eigen::SVectorXd::Zero(2); points[1]=Eigen::SVectorXd::Zero(2); points[2]=Eigen::SVectorXd::Zero(2); points[3]=Eigen::SVectorXd::Zero(2);
    points[0](0)=-2; points[0](1)=1; points[1](0)=2; points[1](1)=-1; points[2](0)=0; points[2](1)=-1; points[3](0)=0; points[3](1)=1;
    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxIter,learning_rate,clip,seuil,beta1,beta2,batch_size,mu,factor,RMin,RMax,
        b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        if (std::abs(study["finalGradient"].error)>eps)
        {
            std::cout << i << ": " << study["finalGradient"].digits() << std::endl;
        }

        if (study["finalGradient"]+std::abs(study["finalGradient"].error)<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée" << std::endl; farMin++;}
            else{iters[numeroPoint]+=study["iter"];}
        }

        else
        {
            nonMin++;
            std::cout << "On n'est pas tombé sur un minimum" << i << std::endl;

            if(Sstd::abs(study["finalGradient"])>1 || Sstd::isnan(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence ou précision: " << i << std::endl;
                numeroPoint = -2;
            }
        }

        if(record)
        {
            gradientNormFlux << numeroPoint << std::endl;
            gradientNormFlux << study["finalGradient"].number << std::endl;
            gradientNormFlux << study["finalGradient"].error << std::endl;

            iterFlux << numeroPoint << std::endl;
            iterFlux << study["iter"].number << std::endl;

            initFlux << numeroPoint << std::endl;
            initFlux << weightsInit[0](0,0).number << std::endl;
            initFlux << biasInit[0](0).number << std::endl;
        }

        if(tracking)
        {
            trackingFlux << numeroPoint << std::endl;
            trackingFlux << study["iter"].number << std::endl;
            trackingFlux << study["prop_entropie"].number << std::endl;
            trackingFlux << study["prop_initial_ineq"].number << std::endl;
        }

        if(track_continuous)
        {
            trackContinuousFlux << numeroPoint << std::endl;
            trackContinuousFlux << study["iter"].number << std::endl;
            trackContinuousFlux << study["continuous_entropie"].number << std::endl;
        }

    }

    for(i=0;i<4;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i]; iters[i]/=proportions[i];}
        proportions[i]/=Sdouble(nbTirage);
    }

    std::cout << "La proportion pour (-2,1): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (-2,1): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,1): " << iters[0] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,-1): " << proportions[1] << std::endl;
    std::cout << "La distance moyenne à (2,-1): " << distances[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,-1): " << iters[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,-1): " << proportions[2] << std::endl;
    std::cout << "La distance moyenne à (0,-1): " << distances[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[2]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,1): " << proportions[3] << std::endl;
    std::cout << "La distance moyenne à (0,1): " << distances[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,1): " << iters[3]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << Sdouble(nonMin)/Sdouble(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << Sdouble(farMin)/Sdouble(nbTirage) << std::endl;

}

void test_PolyFive(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor, Sdouble const& Rlim, Sdouble const& RMin,
Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas,
Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{

    std::ofstream gradientNormFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"iter"+".csv").c_str());
    std::ofstream initFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream trackingFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"tracking"+".csv").c_str());
    std::ofstream trackContinuousFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"track_continuous"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux || !trackingFlux || !trackContinuousFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::SMatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    std::vector<Eigen::SMatrixXd> weightsInit(L);
    std::vector<Eigen::SVectorXd> biasInit(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyFive";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    int i; unsigned seed;
    std::map<std::string,Sdouble> study;
    Eigen::SVectorXd currentPoint(2);
    std::vector<Eigen::SVectorXd> points(6);
    std::vector<Sdouble> proportions(6,0.0), distances(6,0.0), iters(6,0.0);
    int numeroPoint, farMin=0, nonMin=0;

    points[0]=Eigen::SVectorXd::Zero(2); points[1]=Eigen::SVectorXd::Zero(2); points[2]=Eigen::SVectorXd::Zero(2); points[3]=Eigen::SVectorXd::Zero(2);
    points[4]=Eigen::SVectorXd::Zero(2); points[5]=Eigen::SVectorXd::Zero(2);
    points[0](0)=2; points[0](1)=1; points[1](0)=0; points[1](1)=-1; points[2](0)=-2; points[2](1)=3; points[3](0)=0; points[3](1)=3;
    points[4](0)=-4; points[4](1)=3; points[5](0)=4; points[5](1)=-1;

    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxIter,learning_rate,clip,seuil,beta1,beta2,batch_size,mu,factor,RMin,RMax,
        b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        if (std::abs(study["finalGradient"].error)>eps)
        {
            std::cout << i << ": " << study["finalGradient"].digits() << std::endl;
        }

        if (study["finalGradient"]+std::abs(study["finalGradient"].error)<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée: " << i << std::endl; farMin++;}
            else{iters[numeroPoint]+=study["iter"];}
        }

        else
        {
            nonMin++;
            std::cout << "On n'est pas tombé sur un minimum: " << i << std::endl;

            if(Sstd::abs(study["finalGradient"])>1 || Sstd::isnan(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence ou précision: " << i << std::endl;
                numeroPoint = -2;
            }
        }

        if(record)
        {
            gradientNormFlux << numeroPoint << std::endl;
            gradientNormFlux << study["finalGradient"].number << std::endl;
            gradientNormFlux << study["finalGradient"].error << std::endl;

            iterFlux << numeroPoint << std::endl;
            iterFlux << study["iter"].number << std::endl;

            initFlux << numeroPoint << std::endl;
            initFlux << weightsInit[0](0,0).number << std::endl;
            initFlux << biasInit[0](0).number << std::endl;
        }

        if(tracking)
        {
            trackingFlux << numeroPoint << std::endl;
            trackingFlux << study["iter"].number << std::endl;
            trackingFlux << study["prop_entropie"].number << std::endl;
            trackingFlux << study["prop_initial_ineq"].number << std::endl;
        }

        if(track_continuous)
        {
            trackContinuousFlux << numeroPoint << std::endl;
            trackContinuousFlux << study["iter"].number << std::endl;
            trackContinuousFlux << study["continuous_entropie"].number << std::endl;
        }

    }

    for(i=0;i<6;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i]; iters[i]/=proportions[i];}
        proportions[i]/=Sdouble(nbTirage);
    }

    std::cout << "La proportion pour (2,1): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (2,1): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,1): " << iters[0]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,-1): " << proportions[1] << std::endl;
    std::cout << "La distance moyenne à (0,-1): " << distances[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (-2,3): " << proportions[2] << std::endl;
    std::cout << "La distance moyenne à (-2,3): " << distances[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,3): " << iters[2]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,3): " << proportions[3] << std::endl;
    std::cout << "La distance moyenne à (0,3): " << distances[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,3): " << iters[3]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (-4,3): " << proportions[4] << std::endl;
    std::cout << "La distance moyenne à (-4,3): " << distances[4] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-4,3): " << iters[4]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (4,-1): " << proportions[5] << std::endl;
    std::cout << "La distance moyenne à (4,-1): " << distances[5] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (4,-1): " << iters[5] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << Sdouble(nonMin)/Sdouble(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << Sdouble(farMin)/Sdouble(nbTirage) << std::endl;

}

void test_PolyEight(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, Sdouble& mu, Sdouble& factor, Sdouble const& Rlim, Sdouble const& RMin,
Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas,
Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{

    std::ofstream gradientNormFlux(("Record/polyEight/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/polyEight/"+setHyperparameters+"/"+algo+"_"+"iter"+".csv").c_str());
    std::ofstream initFlux(("Record/polyEight/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream trackingFlux(("Record/polyEight/"+setHyperparameters+"/"+algo+"_"+"tracking"+".csv").c_str());
    std::ofstream trackContinuousFlux(("Record/polyEight/"+setHyperparameters+"/"+algo+"_"+"track_continuous"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux || !trackingFlux || !trackContinuousFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::SMatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    std::vector<Eigen::SMatrixXd> weightsInit(L);
    std::vector<Eigen::SVectorXd> biasInit(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyEight";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    int i; unsigned seed;
    std::map<std::string,Sdouble> study;
    Eigen::SVectorXd currentPoint(2);
    std::vector<Eigen::SVectorXd> points(6);
    std::vector<Sdouble> proportions(6,0.0), distances(6,0.0), iters(6,0.0);
    int numeroPoint, farMin=0, nonMin=0;

    points[0]=Eigen::SVectorXd::Zero(2); points[1]=Eigen::SVectorXd::Zero(2); points[2]=Eigen::SVectorXd::Zero(2); points[3]=Eigen::SVectorXd::Zero(2);
    points[4]=Eigen::SVectorXd::Zero(2); points[5]=Eigen::SVectorXd::Zero(2);
    points[0](0)=0; points[0](1)=0; points[1](0)=0; points[1](1)=2; points[2](0)=2; points[2](1)=0; points[3](0)=-2; points[3](1)=2;
    points[4](0)=1; points[4](1)=0; points[5](0)=-1; points[5](1)=1;

    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxIter,learning_rate,clip,seuil,beta1,beta2,batch_size,mu,factor,RMin,RMax,
        b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        if (std::abs(study["finalGradient"].error)>eps)
        {
            std::cout << i << ": " << study["finalGradient"].digits() << std::endl;
        }

        if (study["finalGradient"]+std::abs(study["finalGradient"].error)<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint==-1){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée: " << i << std::endl; farMin++;}
            else{iters[numeroPoint]+=study["iter"];}
        }

        else
        {
            nonMin++;
            std::cout << "On n'est pas tombé sur un minimum: " << i << std::endl;

            if(Sstd::abs(study["finalGradient"])>1 || Sstd::isnan(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence ou précision: " << i << std::endl;
                numeroPoint = -2;
            }
        }

        if(record)
        {
            gradientNormFlux << numeroPoint << std::endl;
            gradientNormFlux << study["finalGradient"].number << std::endl;
            gradientNormFlux << study["finalGradient"].error << std::endl;

            iterFlux << numeroPoint << std::endl;
            iterFlux << study["iter"].number << std::endl;

            initFlux << numeroPoint << std::endl;
            initFlux << weightsInit[0](0,0).number << std::endl;
            initFlux << biasInit[0](0).number << std::endl;
        }

        if(tracking)
        {
            trackingFlux << numeroPoint << std::endl;
            trackingFlux << study["iter"].number << std::endl;
            trackingFlux << study["prop_entropie"].number << std::endl;
            trackingFlux << study["prop_initial_ineq"].number << std::endl;
        }

        if(track_continuous)
        {
            trackContinuousFlux << numeroPoint << std::endl;
            trackContinuousFlux << study["iter"].number << std::endl;
            trackContinuousFlux << study["continuous_entropie"].number << std::endl;
        }

    }

    for(i=0;i<6;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i]; iters[i]/=proportions[i];}
        proportions[i]/=Sdouble(nbTirage);
    }

    std::cout << "La proportion pour (0,0): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (0,0): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,0): " << iters[0]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,2): " << proportions[1] << std::endl;
    std::cout << "La distance moyenne à (0,2): " << distances[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,2): " << iters[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,0): " << proportions[2] << std::endl;
    std::cout << "La distance moyenne à (2,0): " << distances[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,0): " << iters[2]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (-2,2): " << proportions[3] << std::endl;
    std::cout << "La distance moyenne à (-2,2): " << distances[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,2): " << iters[3]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (1,0): " << proportions[4] << std::endl;
    std::cout << "La distance moyenne à (1,0): " << distances[4] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (1,0): " << iters[4]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (-1,1): " << proportions[5] << std::endl;
    std::cout << "La distance moyenne à (-1,1): " << distances[5] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-1,1): " << iters[5] << std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << Sdouble(nonMin)/Sdouble(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << Sdouble(farMin)/Sdouble(nbTirage) << std::endl;

}


void test_Cloche(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor, Sdouble const& Rlim, Sdouble const& RMin,
Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas,
Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{
    std::ofstream gradientNormFlux(("Record/cloche/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/cloche/"+setHyperparameters+"/"+algo+"_"+"iter"+".csv").c_str());
    std::ofstream initFlux(("Record/cloche/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream trackingFlux(("Record/cloche/"+setHyperparameters+"/"+algo+"_"+"tracking"+".csv").c_str());
    std::ofstream trackContinuousFlux(("Record/cloche/"+setHyperparameters+"/"+algo+"_"+"track_continuous"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux || !trackingFlux || !trackContinuousFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::SMatrixXd X(1,3), Y(1,3);
    X(0,0)=0; X(0,1)=1; X(0,2)=2; Y(0,0)=1; Y(0,1)=0; Y(0,2)=1;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    std::vector<Eigen::SMatrixXd> weightsInit(L);
    std::vector<Eigen::SVectorXd> biasInit(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="cloche";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    int i; unsigned seed;
    std::map<std::string,Sdouble> study;
    Eigen::SVectorXd currentPoint(2);
    std::vector<Eigen::SVectorXd> points(2);
    std::vector<Sdouble> proportions(2,0.0), distances(2,0.0), iters(2,0.0);
    int numeroPoint, nonMin=0, farMin=0;

    points[0]=Eigen::SVectorXd::Zero(2); points[1]=Eigen::SVectorXd::Zero(2);
    points[0](0)=0; points[0](1)=-Sstd::sqrt(2*Sstd::log(Sdouble(3.0/2.0))); points[1](0)=0; points[1](1)=Sstd::sqrt(2*Sstd::log(Sdouble(3.0/2.0)));

    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"entropie_one",famille_algo,algo,eps,maxIter,learning_rate,clip,seuil,beta1,beta2,batch_size,mu,factor,RMin,RMax,
        b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        if (study["finalGradient"]+std::abs(study["finalGradient"].error)<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint==-1){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée: " << i << std::endl; farMin++;}
            else{iters[numeroPoint]+=study["iter"];}
        }
        else
        {
            nonMin++;
            std::cout << "On n'est pas tombé sur un minimum: " << i << std::endl;

            if(Sstd::abs(study["finalGradient"])>1 || Sstd::isnan(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence ou précision: " << i << std::endl;
                numeroPoint = -2;
                std::cout << study["finalGradient"] << std::endl;
            }
        }

        if(record)
        {
            gradientNormFlux << numeroPoint << std::endl;
            gradientNormFlux << study["finalGradient"].number << std::endl;
            gradientNormFlux << study["finalGradient"].error << std::endl;

            iterFlux << numeroPoint << std::endl;
            iterFlux << study["iter"].number << std::endl;

            initFlux << numeroPoint << std::endl;
            initFlux << weightsInit[0](0,0).number << std::endl;
            initFlux << biasInit[0](0).number << std::endl;
        }

        if(tracking)
        {
            trackingFlux << numeroPoint << std::endl;
            trackingFlux << study["iter"].number << std::endl;
            trackingFlux << study["prop_entropie"].number << std::endl;
        }

        if(track_continuous)
        {
            trackContinuousFlux << numeroPoint << std::endl;
            trackContinuousFlux << study["iter"].number << std::endl;
            trackContinuousFlux << study["continuous_entropie"].number << std::endl;
            trackingFlux << study["prop_initial_ineq"].number << std::endl;
        }

    }

    for(i=0;i<2;i++)
    {
        if (proportions[i]!=0){distances[i]/=proportions[i]; iters[i]/=proportions[i];}
        proportions[i]/=Sdouble(nbTirage);
    }

    std::cout << "La proportion pour (0,-z0): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (0,-z0): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-z0): " << iters[0]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,z0): " << proportions[1] << std::endl;
    std::cout << "La distance moyenne à (0,z0): " << distances[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,z0): " << iters[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << Sdouble(nonMin)/Sdouble(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << Sdouble(farMin)/Sdouble(nbTirage) << std::endl;
}

void test_RatTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor, Sdouble const& Rlim, Sdouble const& RMin,
Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& alpha, Sdouble const& pas,
Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{
    std::ofstream gradientNormFlux(("Record/ratTwo/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/ratTwo/"+setHyperparameters+"/"+algo+"_"+"iter"+".csv").c_str());
    std::ofstream initFlux(("Record/ratTwo/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream trackingFlux(("Record/ratTwo/"+setHyperparameters+"/"+algo+"_"+"tracking"+".csv").c_str());
    std::ofstream trackContinuousFlux(("Record/ratTwo/"+setHyperparameters+"/"+algo+"_"+"track_continuous"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux || !trackingFlux || !trackContinuousFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::SMatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=1; Y(0,1)=1;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    std::vector<Eigen::SMatrixXd> weightsInit(L);
    std::vector<Eigen::SVectorXd> biasInit(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="ratTwo";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    int i; unsigned seed;
    std::map<std::string,Sdouble> study;
    Eigen::SVectorXd currentPoint(2);
    std::vector<Eigen::SVectorXd> points(1);
    std::vector<Sdouble> proportions(1,0.0), distances(1,0.0), iters(1,0.0);
    int numeroPoint, nonMin=0, farMin=0;

    points[0]=Eigen::SVectorXd::Zero(2);
    points[0](0)=0; points[0](1)=(1+std::sqrt(5))/2;

    for(i=0;i<nbTirage;i++)
    {
        seed=i; initialisation(nbNeurons,weights,bias,supParameters,distribution,seed);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"entropie_one",famille_algo,algo,eps,maxIter,learning_rate,clip,seuil,beta1,beta2,batch_size,mu,factor,RMin,RMax,
        b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        if (study["finalGradient"]+std::abs(study["finalGradient"].error)<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
            if(numeroPoint==-1){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée: " << i << std::endl; farMin++;}
            else{iters[numeroPoint]+=study["iter"];}
        }
        else
        {
            nonMin++;
            std::cout << "On n'est pas tombé sur un minimum: " << i << std::endl;

            if(Sstd::abs(study["finalGradient"])>1 || Sstd::isnan(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence ou précision: " << i << std::endl;
                numeroPoint = -2;
                std::cout << study["finalGradient"] << std::endl;
            }
        }

        if(record)
        {
            gradientNormFlux << numeroPoint << std::endl;
            gradientNormFlux << study["finalGradient"].number << std::endl;
            gradientNormFlux << study["finalGradient"].error << std::endl;

            iterFlux << numeroPoint << std::endl;
            iterFlux << study["iter"].number << std::endl;

            initFlux << numeroPoint << std::endl;
            initFlux << weightsInit[0](0,0).number << std::endl;
            initFlux << biasInit[0](0).number << std::endl;
        }

        if(tracking)
        {
            trackingFlux << numeroPoint << std::endl;
            trackingFlux << study["iter"].number << std::endl;
            trackingFlux << study["prop_entropie"].number << std::endl;
            trackingFlux << study["prop_initial_ineq"].number << std::endl;
        }

        if(track_continuous)
        {
            trackContinuousFlux << numeroPoint << std::endl;
            trackContinuousFlux << study["iter"].number << std::endl;
            trackContinuousFlux << study["continuous_entropie"].number << std::endl;
        }


    }

    if (proportions[0]!=0){distances[0]/=proportions[0]; iters[0]/=proportions[0];}
    proportions[0]/=Sdouble(nbTirage);

    std::cout << "La proportion pour (0,z2): " << proportions[0] << std::endl;
    std::cout << "La distance moyenne à (0,z2): " << distances[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,z2): " << iters[0]<< std::endl;
    std::cout << "" << std::endl;


    std::cout << "Proportion de fois où la condition sur le gradient n'est pas respectée: " << Sdouble(nonMin)/Sdouble(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << Sdouble(farMin)/Sdouble(nbTirage) << std::endl;
}





