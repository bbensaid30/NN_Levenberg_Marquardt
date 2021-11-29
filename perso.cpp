#include "perso.h"

std::map<std::string,Sdouble> EulerRichardson(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& seuil, Sdouble const& eps, int const& maxIter, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_EulerRichardson_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
     std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd gradientInter = Eigen::SVectorXd::Zero(N);


    Sdouble gradientNorm = 1000;
    Sdouble learning_rate = learning_rate_init;
    Sdouble erreur;

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

       if(iter==0)
        {
            std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

            fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(X,Y,L,P,nbNeurons,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
            fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
            backward(X,Y,L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);
        }

        erreur=(0.5*learning_rate*(gradient-gradientInter).norm())/seuil;

        if(erreur>1)
        {
           learning_rate*=0.9/Sstd::sqrt(erreur);
        }
        else
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradientInter);
            learning_rate*=0.9/Sstd::sqrt(erreur);

            fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(X,Y,L,P,nbNeurons,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
        update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
        fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        backward(X,Y,L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);

        gradientNorm = gradientInter.norm();

        if(numericalNoise(gradientNorm)){break;}

        iter++;
    }

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes);
    Sdouble cost = risk(Y,P,As[L],type_perte);
    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost;

    return study;

}

std::map<std::string,Sdouble> Momentum_ER(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate_init, Sdouble const& beta1_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_ER_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
     std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd gradientInter = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1Inter = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm = 1000;
    Sdouble learning_rate = learning_rate_init;
    Sdouble beta1 = beta1_init;
    Sdouble erreur;
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

       if(iter==0)
        {
            std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

            fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(X,Y,L,P,nbNeurons,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            moment1Inter = beta1*gradient;

            update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*moment1Inter);
            fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
            backward(X,Y,L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);
        }

        erreur=(moment1-moment1Inter).squaredNorm()+(moment1+gradientInter-(gradient+moment1Inter)).squaredNorm();
        erreur = 0.5*learning_rate*Sstd::sqrt(erreur)/seuil;

        if(!std::signbit((erreur-1).number))
        {
           learning_rate*=0.9/Sstd::sqrt(erreur);
           beta1*=0.9/Sstd::sqrt(erreur);
        }
        else
        {
            moment1 += -beta1*moment1Inter + beta1*gradientInter;
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1);
            learning_rate*=0.9/Sstd::sqrt(erreur);
            beta1*=0.9/Sstd::sqrt(erreur);

            fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(X,Y,L,P,nbNeurons,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
        update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*moment1);
        fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        backward(X,Y,L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);
        moment1Inter = (1-beta1)*moment1 + beta1*gradient;

        gradientNorm = gradientInter.norm();

        if(numericalNoise(gradientNorm)){std::cout << "Bruit numérique!!!" << std::endl; break;}

        iter++;
    }

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes);
    Sdouble cost = risk(Y,P,As[L],type_perte);
    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost;

    return study;

}

std::map<std::string,Sdouble> LM_ER(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& mu_init,
Sdouble const& seuil, Sdouble const& eps, int const& maxIter, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_LM_ER_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);

     std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);

    Eigen::SVectorXd gradient(N), gradientInter(N), delta(N), deltaInter(N);
    Eigen::SMatrixXd Q(N,N), QInter(N,N);
    Eigen::SMatrixXd I = Eigen::SMatrixXd::Identity(N,N);


    Sdouble gradientNorm = 1000;
    Sdouble h = 1/mu_init;
    Sdouble erreur;

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

       if(iter==0)
        {
            std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

            fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes);
            QSO_backward(X,Y,L,P,nbNeurons,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            solve(gradient,Q+mu_init*I,delta);
            update(L,nbNeurons,globalIndices,weightsInter,biasInter,h*delta);
            fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
            QSO_backward(X,Y,L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,QInter,type_perte);
            solve(gradientInter,QInter+mu_init*I,deltaInter);
        }

        erreur=(0.5*h*(delta-deltaInter).norm())/seuil;

        if(erreur>1)
        {
           h*=0.9/Sstd::sqrt(erreur);
        }
        else
        {
            update(L,nbNeurons,globalIndices,weights,bias,h*deltaInter);
            h*=0.9/Sstd::sqrt(erreur);

            fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes);
            QSO_backward(X,Y,L,P,nbNeurons,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            solve(gradient,Q+mu_init*I,delta);
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

        update(L,nbNeurons,globalIndices,weightsInter,biasInter,h*delta);
        fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        QSO_backward(X,Y,L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,QInter,type_perte);
        solve(gradientInter,QInter+mu_init*I,deltaInter);

        gradientNorm = gradientInter.norm();

        if(numericalNoise(gradientNorm)){break;}

        iter++;
    }

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes);
    Sdouble cost = risk(Y,P,As[L],type_perte);
    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost;

    return study;

}

std::map<std::string,Sdouble> train_Perso(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& learning_rate_init, Sdouble const& beta1_init, Sdouble const& mu_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;

    if(algo=="EulerRichardson")
    {
        study = EulerRichardson(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,seuil,eps,maxIter,record,fileExtension);
    }
    else if(algo=="Momentum_ER")
    {
        study = Momentum_ER(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1_init,seuil,eps,maxIter,record,fileExtension);
    }
    else if(algo=="LM_ER")
    {
        study = LM_ER(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,mu_init,seuil,eps,maxIter,record,fileExtension);
    }

    return study;
}
