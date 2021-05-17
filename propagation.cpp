#include "propagation.h"

//------------------------------------------------------------------Norme2----------------------------------------------------------------------------------------

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

void backwardJacob(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& J)
{
    int l,m,p,n,nL=nbNeurons[L],jump,nbLine=0;
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
            J.row(nbLine) = Jpm;
            nbLine++;
        }
    }

}

//------------------------------------------------------------------Entropie----------------------------------------------------------------------------------------

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

void backwardJacob_entropie(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& E_inv, Eigen::MatrixXd& E2_inv, Eigen::MatrixXd& J, Eigen::MatrixXd& J2)
{
    int l,m,p,n,nL=nbNeurons[L],jump,nbLine=0;
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
            J.row(nbLine) = Jpm;
            Jpm*=E2_inv(m,p);
            J2.row(nbLine) = Jpm;
            nbLine++;
        }
    }

}

//------------------------------------------------------------------Générale----------------------------------------------------------------------------------------

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

void solve(Eigen::VectorXd const& gradient, Eigen::MatrixXd const& hessian, Eigen::VectorXd& delta)
{
    delta = hessian.llt().solve(-gradient);
}

void update(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
Eigen::VectorXd const& delta)
{

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
