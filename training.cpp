#include "training.h"

void fforward(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, int const nbNeurons[], std::string const activations[],
Eigen::MatrixXd weights[], Eigen::VectorXd bias[], Eigen::MatrixXd As[], Eigen::MatrixXd slopes[], Eigen::MatrixXd& E)
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

void backward(int const& L, int const& P, int const nbNeurons[], int const globalIndices[], Eigen::MatrixXd weights[], Eigen::VectorXd bias[],
Eigen::MatrixXd As[], Eigen::MatrixXd slopes[], Eigen::MatrixXd& E, Eigen::VectorXd& gradient, Eigen::MatrixXd& Q)
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
            for (l=L-1;l>0;l--)
            {
                jump=nbNeurons[l+1]*nbNeurons[l];
                dw=dz*(As[l].col(p).transpose());
                dw.resize(jump,1);
                Jpm.segment(globalIndices[2*l]-jump,jump)=dw;
                jump=nbNeurons[l+1];
                Jpm.segment(globalIndices[2*l+1]-jump,jump)=dz;

                dz=(weights[l].transpose()*dz).cwiseProduct(slopes[l-1].col(p));
            }
            Q+=Jpm*Jpm.transpose();
            gradient+=E(m,p)*Jpm;
        }
    }
}

void update(int const& L, int const nbNeurons[], int const globalIndices[], Eigen::MatrixXd weights[], Eigen::VectorXd bias[],
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

void train(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const nbNeurons[], int const globalIndices[], std::string const activations[],
Eigen::MatrixXd weights[], Eigen::VectorXd bias[], double& mu, double& factor, double const& eps, int const& maxIter)
{

    assert (factor>1);

    int N=globalIndices[2*L-1], P=X.cols(), iter=1;

    Eigen::MatrixXd As[L+1]; As[0]=X;
    Eigen::MatrixXd slopes[L];
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N,N);

    Eigen::MatrixXd weightsPrec[L];
    Eigen::MatrixXd biasPrec[L];

    std::copy(weights,weights+L,weightsPrec); std::copy(bias,bias+L,biasPrec);
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    double cost = 0.5*E.squaredNorm(), costPrec;
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
    Q+=mu*I;
    update(L,nbNeurons,globalIndices,weights,bias,gradient,Q);

    while (gradient.norm()>eps || iter<maxIter)
    {
        costPrec=cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost=0.5*E.squaredNorm();

        if (cost<costPrec)
        {
            std::copy(weights,weights+L,weightsPrec); std::copy(bias,bias+L,biasPrec);
            gradient.setZero(); Q.setZero();
            mu/=factor;
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
            Q+=mu*I;
            update(L,nbNeurons,globalIndices,weights,bias,gradient,Q);
        }
        else
        {
            std::copy(weightsPrec,weightsPrec+L,weights); std::copy(biasPrec,biasPrec+L,bias);
            mu*=factor;
        }

        iter++;
    }
    std::cout << gradient.norm() << std::endl;

}
