#include "training_entropie.h"

std::map<std::string,Sdouble> LM_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Sdouble& mu, Sdouble& factor, Sdouble const& eps, int const& maxIter, bool const record, std::string const fileExtension)
{


    std::ofstream weightsFlux(("Record/weights_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    assert (factor>1);

    int N=globalIndices[2*L-1], P=X.cols(), nL=nbNeurons[L], iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E_inv(nbNeurons[L],P), E2_inv(nbNeurons[L],P);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd I = Eigen::SMatrixXd::Identity(N,N);

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv);
    Sdouble cost = entropie(Y,As[L],P,nL), costPrec;
    backward_entropie(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E_inv,E2_inv,gradient,Q);
    H = Q+mu*I;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    Sdouble gradientNorm = gradient.norm();

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        costPrec=cost;
        fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv);
        cost=entropie(Y,As[L],P,nL);

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if (std::signbit((cost-costPrec).number))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            mu/=factor;
            notBack++;
            backward_entropie(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E_inv,E2_inv,gradient,Q);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            mu*=factor;
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu)){break;}

        H = Q+mu*I;
        solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        iter++;
    }

    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv);
    cost=entropie(Y,As[L],P,nL);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=Sdouble(endSequenceMax-notBackMax);
    study["endSequenceMax"]=Sdouble(endSequenceMax); study["startSequenceFinal"]=Sdouble(iter-notBack); study["propBack"]=Sdouble(nbBack)/Sdouble(iter);

    return study;

}

std::map<std::string,Sdouble> LMF_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, Sdouble const& eps, int const& maxIter,
Sdouble const& RMin, Sdouble const& RMax, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), nL=nbNeurons[L], iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E_inv(nL,P), E_invTranspose(P,nL), E2_inv(nL,P);
    Eigen::SMatrixXd intermed(1,P*nL);
    Eigen::SVectorXd gradient(N);
    Eigen::SMatrixXd Q (N,N);
    Eigen::SMatrixXd J(P*nL,N), J2(P*nL,N);
    Eigen::SVectorXd delta(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd D(N,N);
    Sdouble factor, linearReduction, R, mu=10, muc=0, pas=0.1, deriv0;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv); E_invTranspose=E_inv.transpose(); E_invTranspose.resize(P*nL,1);
    Sdouble cost = entropie(Y,As[L],P,nL), costPrec;
    backwardJacob_entropie(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E_inv,E2_inv,J,J2); Q=J2.transpose()*J2; gradient=J.transpose()*E_invTranspose;
    scalingFletcher(Q,D,N);
    H = Q+mu*D;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    Sdouble gradientNorm = gradient.norm();
    Sdouble deltaNorm = delta.norm();

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter && deltaNorm+std::abs(deltaNorm.error)>eps*std::pow(10,-3))
    {
        costPrec = cost;
        fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv); E_invTranspose=E_inv.transpose(); E_invTranspose.resize(P*nL,1);
        cost = entropie(Y,As[L],P,nL);
        intermed=(J*delta).transpose(); intermed.resize(nL,P); linearReduction = costPrec-entropie(Y,As[L]-intermed,P,nL);
        R = (costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if(!std::signbit((R-RMax).number))
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(std::signbit((R-RMin).number))
        {
            deriv0 = (entropie(Y,As[L]-pas*intermed,P,nL)-costPrec)/pas;
            factor = -2*(cost-costPrec-deriv0)/deriv0;
            if(std::signbit((factor-2).number){factor = 2;}
            if(!std::signbit((factor-10).number)){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::SMatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }
        if (std::signbit((cost-costPrec).number))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            notBack++;
            backwardJacob_entropie(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E_inv,E2_inv,J,J2); Q=J2.transpose()*J2; gradient=J.transpose()*E_invTranspose;
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(factor)){break;}

        H = Q+mu*D;
        solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        deltaNorm = delta.norm();
        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv);
    cost = entropie(Y,As[L],P,nL);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=Sdouble(endSequenceMax-notBackMax);
    study["endSequenceMax"]=Sdouble(endSequenceMax); study["startSequenceFinal"]=Sdouble(iter-notBack); study["propBack"]=Sdouble(nbBack)/Sdouble(iter);

    return study;

}

std::map<std::string,Sdouble> LMUphill_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, Sdouble const& eps, int const& maxIter,
Sdouble const& RMin, Sdouble const& RMax, int const& b, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), nL=nbNeurons[L], iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E_inv(nL,P), E_invTranspose(P,nL), E2_inv(nL,P);
    Eigen::SMatrixXd intermed(1,P*nL);
    Eigen::SVectorXd gradient(N);
    Eigen::SMatrixXd Q (N,N);
    Eigen::SMatrixXd J(P*nL,N), J2(P*nL,N);
    Eigen::SVectorXd delta(N), deltaPrec(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd D(N,N);
    Sdouble factor, linearReduction, R, mu=10, muc=0, pas=0.1, deriv0;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv); E_invTranspose=E_inv.transpose(); E_invTranspose.resize(P*nL,1);
    Sdouble cost = entropie(Y,As[L],P,nL), costPrec; Sdouble costMin=cost;
    backwardJacob_entropie(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E_inv,E2_inv,J,J2); Q=J2.transpose()*J2; gradient=J.transpose()*E_invTranspose;
    scalingFletcher(Q,D,N);
    H = Q+mu*D;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    Sdouble gradientNorm = gradient.norm();
    Sdouble deltaNorm = delta.norm();
    Sdouble angle;

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter && deltaNorm+std::abs(deltaNorm.error)>eps*std::pow(10,-3))
    {
        costPrec = cost;
        fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv); E_invTranspose=E_inv.transpose(); E_invTranspose.resize(P*nL,1);
        cost = entropie(Y,As[L],P,nL);
        intermed=(J*delta).transpose(); intermed.resize(nL,P); linearReduction = costPrec-entropie(Y,As[L]-intermed,P,nL);
        R = (costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if(!std::signbit((R-RMax).number))
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(std::signbit((R-RMin).number))
        {
            deriv0 = (entropie(Y,As[L]-pas*intermed,P,nL)-costPrec)/pas;
            factor = -2*(cost-costPrec-deriv0)/deriv0;
            if(std::signbit((factor-2).number)){factor = 2;}
            if(!std::signbit((factor-10).number)){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::SMatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }
        if (std::signbit((cost-costPrec).number)  || (iter>1 && std::signbit((angle-costMin).number)))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            notBack++;
            costMin=minimum(costMin,cost);
            backwardJacob_entropie(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E_inv,E2_inv,J,J2); Q=J2.transpose()*J2; gradient=J.transpose()*E_invTranspose;
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu)`|| numericalNoise(factor)){break;}

        H = Q+mu*D;
        solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        deltaNorm = delta.norm();
        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv);
    cost = entropie(Y,As[L],P,nL);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=Sdouble(endSequenceMax-notBackMax);
    study["endSequenceMax"]=Sdouble(endSequenceMax); study["startSequenceFinal"]=Sdouble(iter-notBack); study["propBack"]=Sdouble(nbBack)/Sdouble(iter);

    return study;

}


std::map<std::string,Sdouble> LMNielson_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Sdouble const& eps, int const& maxIter, Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& epsDiag,
bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LMNielson_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMNielson_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMNielson_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, nL=nbNeurons[L], l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E_inv(nL,P), E_invTranspose(P,nL), E2_inv(nL,P);
    Eigen::SMatrixXd intermed(1,P*nL);
    Eigen::SVectorXd gradient(N);
    Eigen::SMatrixXd Q (N,N);
    Eigen::SMatrixXd J(P*nL,N), J2(P*nL,N);
    Eigen::SVectorXd delta(N), deltaPrec(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd I  = Eigen::SMatrixXd::Identity(N,N);
    Sdouble mu=10, linearReduction, R, nu=beta;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv); E_invTranspose=E_inv.transpose(); E_invTranspose.resize(P*nL,1);
    Sdouble cost = entropie(Y,As[L],P,nL), costPrec;
    backwardJacob_entropie(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E_inv,E2_inv,J,J2); Q=J2.transpose()*J2; gradient=J.transpose()*E_invTranspose;
    mu=tau*Q.diagonal().maxCoeff();
    H = Q+mu*I;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    Sdouble gradientNorm = gradient.norm();

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        costPrec = cost;
        fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv); E_invTranspose=E_inv.transpose(); E_invTranspose.resize(P*nL,1);
        cost = entropie(Y,As[L],P,nL);
        intermed=(J*delta).transpose(); intermed.resize(nL,P); linearReduction = costPrec-entropie(Y,As[L]-intermed,P,nL);
        R = (costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if (!std::signbit(R.number))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            notBack++;
            mu*=maximum(1.0/gamma,1-(beta-1)*Sstd::pow(2*R-1,p));
            backwardJacob_entropie(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E_inv,E2_inv,J,J2); Q=J2.transpose()*J2; gradient=J.transpose()*E_invTranspose;
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
            mu*=nu; nu*=2;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(nu)){break;}

        H = Q+mu*I;
        solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv);
    cost = entropie(Y,As[L],P,nL);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=Sdouble(endSequenceMax-notBackMax);
    study["endSequenceMax"]=Sdouble(endSequenceMax); study["startSequenceFinal"]=Sdouble(iter-notBack); study["propBack"]=Sdouble(nbBack)/Sdouble(iter);

    return study;
}

std::map<std::string,Sdouble> train_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& algo, Sdouble const& eps, int const& maxIter,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& epsDiag, Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p,
Sdouble const& sigma, bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;
    if(algo=="LM"){study = LM_entropie(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter,record,fileExtension);}
    else if(algo=="LMF"){study = LMF_entropie(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,record,fileExtension);}
    else if(algo=="LMUphill"){study = LMUphill_entropie(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b,record,fileExtension);}
    else if(algo=="LMNielson"){study = LMNielson_entropie(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,tau,beta,gamma,p,epsDiag,record,fileExtension);}

    return study;
}
