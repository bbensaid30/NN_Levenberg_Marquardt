#include "training_entropie.h"

std::map<std::string,double> LM_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double mu, double factor, double const eps, int const maxIter, bool record, std::string fileExtension)
{


    std::ofstream weightsFlux(("Record/weights_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    assert (factor>1);

    int N=globalIndices[2*L-1], P=X.cols(), nL=nbNeurons[L], iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E_inv(nbNeurons[L],P), E2_inv(nbNeurons[L],P);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N,N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv);
    double cost = entropie(Y,As[L],P,nL), costPrec;
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
            costFlux << cost << std::endl;
        }
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    while (gradient.norm()>eps && iter<maxIter)
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
                costFlux << cost << std::endl;
            }
        }

        if (cost<costPrec)
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            mu/=factor;
            notBack++;
            backward_entropie(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E_inv,E2_inv,gradient,Q);
            H = Q+mu*I;
            solve(gradient,H,delta);
            update(L,nbNeurons,globalIndices,weights,bias,delta);
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            mu*=factor;
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
            //std::cout << "La valeur intermédiaire de w : " << weights[0] << std::endl;
            //std::cout << "La valeur intermédiaire de b : " << bias[0] << std::endl;
            H = Q+mu*I;
            solve(gradient,H,delta);
            update(L,nbNeurons,globalIndices,weights,bias,delta);
        }

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv);
    cost=entropie(Y,As[L],P,nL);

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;

}

std::map<std::string,double> LMF_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const eps, int const maxIter,
double const RMin, double const RMax, bool const record, std::string fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), nL=nbNeurons[L], iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E_inv(nL,P), E_invTranspose(P,nL), E2_inv(nL,P);
    Eigen::VectorXd gradient(N), Epp(P*nL);
    Eigen::MatrixXd Q (N,N);
    Eigen::MatrixXd J(P*nL,N), J2(P*nL,N);
    Eigen::VectorXd delta(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd D(N,N);
    double factor, linearReduction, R, muc, intermed;
    double mu=0;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv); E_invTranspose=E_inv.transpose(); E_invTranspose.resize(P*nL,1);
    double cost = entropie(Y,As[L],P,nL), costPrec;
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

    while (gradient.norm()>eps && iter<maxIter && delta1.lpNorm<Eigen::Infinity>()>eps*0.0001)
    {
        costPrec = cost;
        fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv); E_invTranspose=E_inv.transpose(); E_invTranspose.resize(P*nL,1);
        cost = entropie(Y,As[L],P,nL);
        linearReduction = costPrec-entropie(Y,As[L]+J*delta,P,nL);
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
        }

        if(R>RMax)
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(R<RMin)
        {
            factor = (cost-costPrec)/(delta.transpose()*gradient)+2;
            if(factor<2){factor = 2;}
            if(factor>10){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }
        if (cost<costPrec)
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            notBack++;
            costMin=std::min(costMin,cost);
            backwardJacob_entropie(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E_inv,E2_inv,J,J2); Q=J2.transpose()*J2; gradient=J.transpose()*E_invTranspose;
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        H = Q+mu*D;
        solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    fforward_entropie(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E_inv,E2_inv);
    cost = entropie(Y,As[L],P,nL);

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;

}

