#include "training_entropie.h"

void fforward_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<Eigen::MatrixXd>& As,
std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E_inv, Eigen::MatrixXd& E2_inv)
{
    int l;
    int nL = nbNeurons[L];
    for (l=0;l<L;l++)
    {
        As[l+1] = weights[l]*As[l];
        As[l+1].colwise() += bias[l];
        activation(activations[l], As[l+1], slopes[l]);
    }

    if(nL==1)
    {
        E_inv = Y.array()/As[L].array() - (1-Y.array())/(1-As[L].array());
        E2_inv = (Y.array()/As[L].array().pow(2) + (1-Y.array())/(1-As[L].array()).pow(2)).sqrt();
    }
    else
    {
         E_inv = Y.array()/As[L].array();
         E2_inv = Y.array().sqrt()/As[L].array();
    }
}

void backward_entropie(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E_inv, Eigen::MatrixXd& E2_inv, Eigen::VectorXd& gradient, Eigen::MatrixXd& Q)
{
    int l,m,p,n,nL=nbNeurons[L],jump;
    int N=globalIndices[2*L-1];

    Eigen::VectorXd dzL(nL);
    Eigen::VectorXd dz;
    Eigen::MatrixXd dw;
    Eigen::VectorXd Jpm(N);

    for (p=0;p<P;p++)
    {
        for (m=0;m<nL;m++)
        {
            for (n=0;n<nL;n++)
            {
                dzL(n) = (n==m) ? -slopes[L-1](m,p) : 0;
            }
            dz=dzL;
            jump=nbNeurons[L]*nbNeurons[L-1];
            dw=dz*(As[L-1].col(p).transpose());
            dw.resize(jump,1);
            Jpm.segment(globalIndices[2*(L-1)]-jump,jump)=dw;
            jump=nbNeurons[L];
            Jpm.segment(globalIndices[2*(L-1)+1]-jump,jump)=dz;
            for (l=L-1;l>0;l--)
            {
                dz=(weights[l].transpose()*dz).cwiseProduct(slopes[l-1].col(p));

                jump=nbNeurons[l]*nbNeurons[l-1];
                dw=dz*(As[l-1].col(p).transpose());
                dw.resize(jump,1);
                Jpm.segment(globalIndices[2*(l-1)]-jump,jump)=dw;
                jump=nbNeurons[l];
                Jpm.segment(globalIndices[2*(l-1)+1]-jump,jump)=dz;

            }
            gradient+=E_inv(m,p)*Jpm;
            Jpm*=E2_inv(m,p);
            Q+=Jpm*Jpm.transpose();
        }
    }
    Q/=P; gradient/=P;

}

void update_entropie(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
Eigen::VectorXd const& gradient, Eigen::MatrixXd const& hessian)
{
    Eigen::VectorXd delta = hessian.llt().solve(-gradient);
    int l, jump;

    for (l=0;l<L;l++)
    {
        jump=nbNeurons[l]*nbNeurons[l+1];
        weights[l].resize(jump,1);
        weights[l] += delta.segment(globalIndices[2*l]-jump,jump);
        weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
        jump=nbNeurons[l+1];
        bias[l] += delta.segment(globalIndices[2*l+1]-jump,jump);
    }

}

double entropie(Eigen::MatrixXd const& Y, Eigen::MatrixXd const& outputs, int const& P, int const& nL)
{
    if(nL==1)
    {
        return (-1.0/(double)P)*(Y.array()*outputs.array().log()+(1-Y.array())*(1-outputs.array()).log()).sum();
    }
    else
    {
        return (-1.0/(double)P)*(Y.array()*outputs.array().log()).sum();
    }
}

std::map<std::string,double> train_entropie(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
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
    update_entropie(L,nbNeurons,globalIndices,weights,bias,gradient,H);

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
            update_entropie(L,nbNeurons,globalIndices,weights,bias,gradient,H);
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
            update_entropie(L,nbNeurons,globalIndices,weights,bias,gradient,H);
        }

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;

}

