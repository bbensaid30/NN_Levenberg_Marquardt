#include "perso.h"

std::map<std::string,Sdouble> PGD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_PGD_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_PGD_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_PGD_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);


    Sdouble V_dot, gradientNorm;
    Sdouble learning_rate = learning_rate_init;

    Sdouble prop_entropie=0, seuilE=0.01;
    Sdouble cost,costPrec,costInter;
    Sdouble lambda=0;
    bool projection;
    Sdouble eps_R = eps/10;
    int iterLoop=0;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte); costPrec=cost;
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    V_dot = gradient.squaredNorm(); gradientNorm = Sstd::sqrt(V_dot);
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

        learning_rate=learning_rate_init;
        do{costInter = cost-learning_rate*V_dot; learning_rate/=1.5;}while(costInter<0);
        //std::cout << "eta: " << learning_rate << std::endl;
        //std::cout << "costInter: " << costInter << std::endl;

        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        cost = risk(Y,P,As[L],type_perte);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        V_dot = gradient.squaredNorm();
        //std::cout << "gradientNorm2: " << V_dot << std::endl;
        lambda=0;
        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        projection = (Sstd::abs(cost-costInter)>eps_R);
        iterLoop=0;
        while(projection && iterLoop<1000)
        {
            lambda -= (cost-costInter)/V_dot;
            //std::cout << "lambda: " << lambda << std::endl;
            //std::cout << "cost :" << cost << std::endl;
            update(L,nbNeurons,globalIndices,weights,bias,lambda*gradient);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            cost = risk(Y,P,As[L],type_perte);
            projection = (Sstd::abs(cost-costInter)>eps_R);
            //std::cout << "projection " << projection << std::endl;
            if(projection){std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());;}
            iterLoop++;
        }

        //std::cout << "convergence: " << cost << std::endl;

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        cost = risk(Y,P,As[L],type_perte);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        V_dot = gradient.squaredNorm(); gradientNorm = Sstd::sqrt(V_dot);

        if(tracking)
        {
            if((cost-costPrec)/costPrec>seuilE){prop_entropie+=1;}
            costPrec=cost;
        }

        iter+=1;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"]=prop_entropie/iter;}

    return study;

}

std::map<std::string,Sdouble> PM(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& beta1_init, Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_PM_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_PM_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_PM_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1=Eigen::SVectorXd::Zero(N), moment1Prec(N);


    Sdouble inv,vSquare=0, arret, gradientNorm2;
    Sdouble learning_rate = learning_rate_init, beta1 = beta1_init;
    Sdouble beta_bar = beta1/learning_rate;

    Sdouble prop_entropie=0, seuilE=0.01;
    Sdouble E,EPrec,EInter;
    Sdouble lambda=0;
    bool projection;
    Sdouble eps_E = eps/10;
    int iterLoop=0;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    E = risk(Y,P,As[L],type_perte); EPrec=E;
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    gradientNorm2 = gradient.squaredNorm(); arret=Sstd::sqrt(gradientNorm2);
    while (arret+std::abs(arret.error)>eps && iter<maxIter)
    {

        learning_rate=learning_rate_init; beta1=beta1_init;
        do{EInter = E-beta1*vSquare; learning_rate/=1.5; beta1/=1.5;}while(EInter<0);
        //std::cout << "eta: " << learning_rate << std::endl;
        //std::cout << "costInter: " << costInter << std::endl;

        update(L,nbNeurons,globalIndices,weights,bias,learning_rate*moment1);
        moment1 = (1-beta1)*moment1-beta1*gradient;

        vSquare=moment1.squaredNorm();
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        E = beta_bar*risk(Y,P,As[L],type_perte)+0.5*vSquare;
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        inv = Sstd::pow(beta_bar,2)*gradient.squaredNorm()+vSquare;
        //std::cout << "gradientNorm2: " << V_dot << std::endl;
        lambda=0;
        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin()); moment1Prec=moment1;
        projection = (Sstd::abs(E-EInter)>eps_E);
        iterLoop=0;
        while(projection && iterLoop<1000)
        {
            lambda -= (E-EInter)/inv;
            //std::cout << "lambda: " << lambda << std::endl;
            //std::cout << "cost :" << cost << std::endl;
            update(L,nbNeurons,globalIndices,weights,bias,lambda*beta_bar*gradient);
            moment1*=(1+lambda);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            E = beta_bar*risk(Y,P,As[L],type_perte)+0.5*vSquare;
            projection = (Sstd::abs(E-EInter)>eps_E);
            //std::cout << "projection " << projection << std::endl;
            if(projection){std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin()); moment1=moment1Prec;}
            iterLoop++;
        }

        //std::cout << "convergence: " << cost << std::endl;

        vSquare=moment1.squaredNorm();
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        E = beta_bar*risk(Y,P,As[L],type_perte)+0.5*vSquare;
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm2=gradient.squaredNorm(); arret=Sstd::sqrt(gradientNorm2+vSquare);

        if(tracking)
        {
            if((E-EPrec)/EPrec>seuilE){prop_entropie+=1;}
            EPrec=E;
        }

        iter+=1;
        if(numericalNoise(gradientNorm2) || numericalNoise(vSquare) ){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=Sstd::sqrt(gradientNorm2); study["finalCost"]=(E-0.5*vSquare)/beta_bar; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"]=prop_entropie/iter;}

    return study;

}


std::map<std::string,Sdouble> PER(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& seuil, Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_PER_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_PER_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_PER_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N), gradientPrec(N);

    Sdouble V_dot, gradientNorm, erreur;
    Sdouble learning_rate = learning_rate_init;

    Sdouble prop_entropie=0, seuilE=0.01;
    Sdouble cost,costPrec,costInter;
    Sdouble lambda=0;
    bool projection;
    Sdouble eps_R = eps/10;
    int iterLoop=0;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte); costPrec=cost;
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    V_dot = gradient.squaredNorm(); gradientNorm = Sstd::sqrt(V_dot);
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin()); gradientPrec=gradient;
        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
        costInter = cost-learning_rate*V_dot;

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        erreur=learning_rate*(gradient-gradientPrec).norm()/(2*seuil);

        if(erreur>1){std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin()); learning_rate*=0.9/Sstd::sqrt(erreur);}
        else
        {
            cost = risk(Y,P,As[L],type_perte);
            V_dot = gradient.squaredNorm();
            //std::cout << "gradientNorm2: " << V_dot << std::endl;
            lambda=0;
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            projection = (Sstd::abs(cost-costInter)>eps_R);
            iterLoop=0;
            while(projection && iterLoop<1000)
            {
                lambda -= (cost-costInter)/V_dot;
                //std::cout << "lambda: " << lambda << std::endl;
                //std::cout << "cost :" << cost << std::endl;
                update(L,nbNeurons,globalIndices,weights,bias,lambda*gradient);
                fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
                cost = risk(Y,P,As[L],type_perte);
                projection = (Sstd::abs(cost-costInter)>eps_R);
                //std::cout << "projection " << projection << std::endl;
                if(projection){std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());}
                iterLoop++;
            }

            //std::cout << "convergence: " << cost << std::endl;

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            cost = risk(Y,P,As[L],type_perte);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            V_dot = gradient.squaredNorm(); gradientNorm = Sstd::sqrt(V_dot);

            learning_rate*=0.9/Sstd::sqrt(erreur);
        }


        if(tracking)
        {
            if((cost-costPrec)/costPrec>seuilE){prop_entropie+=1;}
            costPrec=cost;
        }

        iter+=1;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"]=prop_entropie/iter;}

    return study;

}


std::map<std::string,Sdouble> EulerRichardson(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_EulerRichardson_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_EulerRichardson_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_EulerRichardson_"+fileExtension+".csv").c_str());
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

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
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

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            if(tracking)
            {
                modif++;
                costPrec = cost;
                cost = risk(Y,P,As[L],type_perte);
                if((cost-costPrec)/costPrec>seuilE){prop_entropie++;}
                if(!std::signbit((cost-costInit).number)){prop_initial_ineq++;}

                if(record){speedFlux << ((cost-costPrec)/learning_rate + gradient.squaredNorm()).number << std::endl;}
            }
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            //std::cout << learning_rate << std::endl;
            learning_rate*=0.9/Sstd::sqrt(erreur);
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
        update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
        fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);

        gradientNorm = gradientInter.norm();
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;

        iter++;
        if(numericalNoise(gradientNorm)){break;}

        if(record)
        {
            if(tracking){costsFlux << cost.number << std::endl;}
            gradientNormFlux << gradientNorm.number << std::endl;
        }

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=Sdouble(time);
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
    std::ofstream speedFlux(("Record/speed_GD_Em_"+fileExtension+".csv").c_str());
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
    Sdouble learning_rate = 0.1;
    Sdouble seuilE=0.0, facteur=1.5, factor2=1.5;

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costInit = cost;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
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
        learning_rate=0.1;
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


        if(cost-costPrec>0)
        {
            Em_count+=1;
        }
        if(cost-costInit>0)
        {
            prop_initial_ineq+=1;
        }
        if(record){speedFlux << ((cost-costPrec)/learning_rate + gradient.squaredNorm()).number << std::endl;}

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;
        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
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




//Diminution naive du pas
std::map<std::string,Sdouble> Momentum_Em(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& beta1_init, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_Em_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_Momentum_Em_"+fileExtension+".csv").c_str());
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
    Sdouble learning_rate = 0.1, beta1 = beta1_init;
    Sdouble seuil=0.0, facteur=1.5;

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costInit = cost;
    Em = beta_bar*cost;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
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
        learning_rate=0.1; beta1=beta1_init;
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


        if(Em-EmPrec>0)
        {
            Em_count+=1;
        }
        if(cost-costInit>0)
        {
            prop_initial_ineq+=1;
        }
        if(record){speedFlux << ((Em-EmPrec)/learning_rate+beta_bar*moment1.squaredNorm()).number << std::endl;}

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm=gradient.norm();
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;
        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

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
    else if(algo=="PGD")
    {
        study = PGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="PM")
    {
        study = PM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="PER")
    {
        study = PM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,seuil,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LM_ER")
    {
        study = LM_ER(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,mu_init,seuil,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="GD_Em")
    {
        study = GD_Em(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="Momentum_Em")
    {
        study = Momentum_Em(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }


    return study;
}
