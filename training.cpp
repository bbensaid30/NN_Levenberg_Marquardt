#include "training.h"

void fforward(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E)
{
    int l;
    for (l=0;l<L;l++)
    {
        As[l+1] = weights[l]*As[l];
        As[l+1].colwise() += bias[l];
        activation(activations[l], As[l+1], slopes[l]);
    }
    E=Y-As[L];
}

void backward(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E, Eigen::VectorXd& gradient, Eigen::MatrixXd& Q)
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
            Q+=Jpm*Jpm.transpose();
            gradient+=E(m,p)*Jpm;
        }
    }

}

void update(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
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

std::map<std::string,double> train(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double mu, double factor, double const eps, int const maxIter)
{

    assert (factor>1);

    int N=globalIndices[2*L-1], P=X.cols(), iter=1;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N,N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    double cost = 0.5*E.squaredNorm(), costPrec;
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
    H = Q+mu*I;
    update(L,nbNeurons,globalIndices,weights,bias,gradient,H);

    while (gradient.norm()>eps && iter<maxIter)
    {
        costPrec=cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost=0.5*E.squaredNorm();

        if (cost<costPrec)
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            mu/=factor;
            notBack++;
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
            H = Q+mu*I;
            update(L,nbNeurons,globalIndices,weights,bias,gradient,H);
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
            update(L,nbNeurons,globalIndices,weights,bias,gradient,H);
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
