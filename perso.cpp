#include "perso.h"

std::map<std::string,Sdouble> EulerRichardson(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_EulerRichardson_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_EulerRichardson_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

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

    Sdouble prop_entropie=0, prop_initial_ineq=0, modif=0, seuilE=0.01;
    Sdouble costInit,cost,costPrec;

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

       if(iter==0)
        {
            std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            if(tracking){cost = risk(Y,P,As[L],type_perte); costInit = cost;}

            if(record)
            {
                if(tracking){costsFlux << cost.number << std::endl;}
                gradientNorm = gradient.norm(); gradientNormFlux << gradientNorm.number << std::endl;
            }

            update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);
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

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            if(tracking)
            {
                modif++;
                costPrec = cost;
                cost = risk(Y,P,As[L],type_perte);
                if((cost-costPrec)/costPrec>seuilE){prop_entropie++;}
                if(!std::signbit((cost-costInit).number)){prop_initial_ineq++;}
            }
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
        update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
        fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);

        gradientNorm = gradientInter.norm();

        iter++;
        if(numericalNoise(gradientNorm)){break;}

        if(record)
        {
            if(tracking){costsFlux << cost.number << std::endl;}
            gradientNormFlux << gradientNorm.number << std::endl;
        }

    }

    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost;
    if(tracking){study["prop_entropie"]=prop_entropie/modif; study["prop_initial_ineq"]=prop_initial_ineq/modif;}

    return study;

}

std::map<std::string,Sdouble> ER_Em(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_ER_Em_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_ER_Em_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd gradientInter = Eigen::SVectorXd::Zero(N);


    Sdouble gradientNorm = 1000;
    Sdouble learning_rate = learning_rate_init, learning_rate_ER;
    Sdouble erreur, facteur=1.5;

    Sdouble prop_entropie=0, prop_initial_ineq=0, modif=0, seuilE=0.01;
    Sdouble costInit,cost,costPrec;

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

       if(iter==0)
        {
            std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            cost = risk(Y,P,As[L],type_perte); costInit = cost;

            if(record)
            {
                if(tracking){costsFlux << cost.number << std::endl;}
                gradientNorm = gradient.norm(); gradientNormFlux << gradientNorm.number << std::endl;
            }

            update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);
        }

        erreur=(0.5*learning_rate*(gradient-gradientInter).norm())/seuil;

        if(erreur>1)
        {
           learning_rate*=0.9/Sstd::sqrt(erreur);
        }
        else
        {
            learning_rate_ER = learning_rate;
            costPrec=cost;
            std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
            do
            {
                update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradientInter);
                fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
                if((cost-costPrec)/costPrec>seuilE){learning_rate/=facteur;std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());}
            }while((cost-costPrec)/costPrec>seuilE);

            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            //learning_rate*=0.9/Sstd::sqrt(erreur);
            learning_rate = (0.9*learning_rate_ER)/Sstd::sqrt(erreur);

            if(tracking)
            {
                modif++;
                if((cost-costPrec)/costPrec>seuilE){prop_entropie++;}
                if(!std::signbit((cost-costInit).number)){prop_initial_ineq++;}
            }
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
        update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
        fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);

        gradientNorm = gradientInter.norm();

        iter++;
        if(numericalNoise(gradientNorm)){break;}

        if(record)
        {
            if(tracking){costsFlux << cost.number << std::endl;}
            gradientNormFlux << gradientNorm.number << std::endl;
        }

    }

    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost;
    if(tracking){study["prop_entropie"]=prop_entropie/modif; study["prop_initial_ineq"]=prop_initial_ineq/modif;}

    return study;

}

//Diminution naive du pas pour GD
std::map<std::string,Sdouble> GD_Em(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_GD_Em_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterTot=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm=1000;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    Sdouble cost,costPrec,costInit;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble learning_rate = learning_rate_init;
    Sdouble seuilE=0.0, facteur=1.5, factor2=1.5;

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costInit = cost;
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

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec=cost;
        learning_rate=learning_rate_init;
        do
        {
            //iterTot++;
            //if(iterTot<10){std::cout << learning_rate << std::endl;}
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);

            cost = risk(Y,P,As[L],type_perte);
            //std::cout << Em << std::endl;
            if(cost-costPrec>0){learning_rate/=facteur;std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());}
        }while(cost-costPrec>0);
        //std::cout << "learning_rate: " << learning_rate << std::endl;

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }
        if(cost-costInit>0)
        {
            prop_initial_ineq+=1;
        }

        gradientNorm=gradient.norm();
        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> LM_ER(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& mu_init,
Sdouble const& seuil, Sdouble const& eps, int const& maxIter, bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_LM_ER_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    Sdouble prop_entropie=0, prop_initial_ineq=0, modif=0;
    Sdouble costInit, cost, costPrec;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);

     std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);

    Eigen::SVectorXd gradient(N), gradientInter(N), delta(N), deltaInter(N);
    Eigen::SMatrixXd Q(N,N), QInter(N,N);
    Eigen::SMatrixXd I = Eigen::SMatrixXd::Identity(N,N);

    Sdouble gradientNorm = 1000;
    Sdouble h = 1/mu_init;
    Sdouble mu = mu_init;
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

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            solve(gradient,Q+mu*I,delta);
            if(tracking){cost = risk(Y,P,As[L],type_perte); costInit = cost;}

            update(L,nbNeurons,globalIndices,weightsInter,biasInter,h*delta);
            fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,QInter,type_perte);
            solve(gradientInter,QInter+mu*I,deltaInter);
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

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            solve(gradient,Q+mu*I,delta);

            if(tracking)
            {
                modif++;
                costPrec = cost;
                cost = risk(Y,P,As[L],type_perte);
                if(std::signbit((cost-costPrec).number)){prop_entropie++;}
                if(std::signbit((cost-costInit).number)){prop_initial_ineq++;}
            }
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

        update(L,nbNeurons,globalIndices,weightsInter,biasInter,h*delta);
        fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,QInter,type_perte);
        solve(gradientInter,QInter+mu*I,deltaInter);


        gradientNorm = gradientInter.norm();

        if(numericalNoise(gradientNorm)){break;}

        iter++;
    }

    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost;
    if(tracking){study["prop_entropie"]=prop_entropie/modif; study["prop_initial_ineq"]=prop_initial_ineq/modif;}

    return study;

}

std::map<std::string,Sdouble> Momentum_Verlet(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
Sdouble const& beta1, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_Verlet_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;
    Sdouble beta_bar = beta1/learning_rate;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N), gradientPrec(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm=1000;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    Sdouble EmPrec,Em,cost,costInit;

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
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            if(tracking)
            {
                cost = risk(Y,P,As[L],type_perte); costInit = cost;
                Em = beta_bar*cost;
            }
        }

        update(L,nbNeurons,globalIndices,weights,bias,learning_rate*moment1-learning_rate*beta1/2*(moment1+gradient));

        gradientPrec=gradient;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        moment1 = ((1-beta1/2)*moment1-beta1/2*(gradientPrec+gradient))/(1+beta1/2);

        gradientNorm = gradient.norm();

        if(tracking)
        {
            cost = risk(Y,P,As[L],type_perte);
            EmPrec = Em; Em = 0.5*moment1.squaredNorm()+beta_bar*cost;
            if(Em-EmPrec>0)
            {
                Em_count+=1;
            }
            if(cost-costInit>0)
            {
                prop_initial_ineq+=1;
            }
        }

        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}



//Diminution naive du pas
std::map<std::string,Sdouble> Momentum_Em(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& beta1_init, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_Em_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterTot=0, l;
    Sdouble beta_bar = beta1_init/learning_rate_init;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N), moment1Prec(N);

    Sdouble gradientNorm=1000;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    Sdouble EmPrec,Em,cost,costInit;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble learning_rate = learning_rate_init, beta1 = beta1_init;
    Sdouble seuil=0.0, facteur=1.5, factor2=1.5;

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costInit = cost;
    Em = beta_bar*cost;
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

        moment1Prec=moment1; std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        EmPrec=Em;
        learning_rate=learning_rate_init; beta1=beta1_init;
        do
        {
            //iterTot++;
            //if(iterTot<10){std::cout << learning_rate << std::endl;}
            moment1 = (1-beta1)*moment1 + beta1*gradient;
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1);

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);

            cost = risk(Y,P,As[L],type_perte);
            Em = 0.5*moment1.squaredNorm()+beta_bar*cost;
            //std::cout << Em << std::endl;
            if(Em-EmPrec>0){learning_rate/=facteur; beta1/=facteur; std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin()); moment1=moment1Prec;}
        }while(Em-EmPrec>0);
        //std::cout << "learning_rate: " << learning_rate << std::endl;

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        if(Em-EmPrec>0)
        {
            Em_count+=1;
        }
        if(cost-costInit>0)
        {
            prop_initial_ineq+=1;
        }

        gradientNorm=gradient.norm();
        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}


std::map<std::string,Sdouble> RK2Momentum(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& beta1_init, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_Em_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;
    Sdouble beta_bar = beta1_init/learning_rate_init;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N), gradientInter = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N), moment1Inter(N);

    Sdouble gradientNorm=1000;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0, seuilE=0.01;
    Sdouble EmPrec,Em,cost,costPrec;

    std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    Sdouble learning_rate = learning_rate_init, beta1=beta1_init;


    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);
    Em = 0.5*moment1.squaredNorm()+beta_bar*cost;
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    do
    {
        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
        update(L,nbNeurons,globalIndices,weights,bias,0.5*learning_rate*(2-beta1)*moment1-0.5*learning_rate*beta1*gradient);

        update(L,nbNeurons,globalIndices,weightsInter,biasInter,learning_rate*moment1);
        fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);
        moment1 += -0.5*beta1*(2-beta1)*moment1 - 0.5*beta1*(1-beta1)*gradient - 0.5*beta1*gradientInter;

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        cost = risk(Y,P,As[L],type_perte);
        EmPrec = Em; Em = 0.5*moment1.squaredNorm()+beta_bar*cost;
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        if((Em-EmPrec)/EmPrec>seuilE){Em_count++;}

        gradientNorm = gradient.norm();

        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }while(gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> Nesterov2(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
Sdouble const& eps, int const& maxIter, bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Nesterov2_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(),l;
    Sdouble iter=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N), gradientY = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm=1000;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    Sdouble EmPrec,Em,cost,costInit;

    std::vector<Eigen::SMatrixXd> weightsY(L), weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasY(L), biasPrec(L);
    std::copy(weights.begin(),weights.end(),weightsY.begin()); std::copy(bias.begin(),bias.end(),biasY.begin());

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
            fforward(L,P,nbNeurons,activations,weightsY,biasY,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weightsY,biasY,As,slopes,gradientY,type_perte);

            if(tracking)
            {
                cost = risk(Y,P,As[L],type_perte); costInit = cost;
                Em = cost;
            }
        }

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        updateNesterov(L,nbNeurons,globalIndices,weights,bias,weightsY,biasY,-learning_rate*gradientY,1,1);
        nesterovCombination(weights,bias,weightsPrec,biasPrec,weightsY,biasY,L,iter/(iter+3));

        fforward(L,P,nbNeurons,activations,weightsY,biasY,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weightsY,biasY,As,slopes,gradientY,type_perte);
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm = gradient.norm();

        if(tracking)
        {
            costInit=cost; cost = risk(Y,P,As[L],type_perte);
            EmPrec=Em; Em = cost+0.5*Sstd::pow(distance(weights,bias,weightsPrec,biasPrec),2);

            if(Em-EmPrec>0)
            {
                Em_count+=1;
            }
            if(cost-costInit>0)
            {
                prop_initial_ineq+=1;
            }
        }


        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> ERIto(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_ERIto_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_ERIto_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    std::normal_distribution<double> d{0,1};

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

    Sdouble prop_entropie=0, prop_initial_ineq=0, modif=0;
    Sdouble costInit,cost,costPrec;

    Sdouble eps1 = std::pow(10,-2);
    while (gradientNorm+std::abs(gradientNorm.error)>eps1 && iter<maxIter)
    {

       if(iter==0)
        {
            std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            if(tracking){cost = risk(Y,P,As[L],type_perte); costInit = cost;}

            if(record)
            {
                gradientNorm = gradient.norm(); gradientNormFlux << gradientNorm.number << std::endl;
            }

            update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);
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

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            if(tracking)
            {
                modif++;
                costPrec = cost;
                cost = risk(Y,P,As[L],type_perte);
                if(std::signbit((cost-costPrec).number)){prop_entropie++;}
                if(std::signbit((cost-costInit).number)){prop_initial_ineq++;}
            }
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
        update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
        fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);

        gradientNorm = gradientInter.norm();

        if(numericalNoise(gradientNorm)){break;}

        iter++;

        if(record)
        {
            gradientNormFlux << gradientNorm.number << std::endl;
        }

    }

     Sdouble r=1;
     learning_rate = 0.01;
     while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        std::mt19937 gen{iter};

        if(record)
        {
            gradientNorm = gradient.norm(); gradientNormFlux << gradientNorm.number << std::endl;
        }

        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient+(std::sqrt(2)+r)*Sstd::sqrt(learning_rate)*d(gen)*gradient);

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm = gradient.norm();
        std::cout << gradientNorm << std::endl;

        if(numericalNoise(gradientNorm)){break;}

        iter++;

    }

    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost;
    if(tracking){study["prop_entropie"]=prop_entropie/modif; study["prop_initial_ineq"]=prop_initial_ineq/modif;}

    return study;

}


std::map<std::string,Sdouble> train_Perso(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& learning_rate_init, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& mu_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;

    if(algo=="EulerRichardson")
    {
        study = EulerRichardson(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,seuil,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="ER_Em")
    {
        study = ER_Em(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,seuil,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="ERIto")
    {
        study = ERIto(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,seuil,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LM_ER")
    {
        study = LM_ER(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,mu_init,seuil,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="Momentum_Verlet")
    {
        study = Momentum_Verlet(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="RK2Momentum")
    {
        study = RK2Momentum(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="GD_Em")
    {
        study = GD_Em(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="Momentum_Em")
    {
        study = Momentum_Em(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="Nesterov2")
    {
         study = Nesterov2(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }


    return study;
}
