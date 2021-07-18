#include "propagation.h"

//------------------------------------------------------------------Norme2----------------------------------------------------------------------------------------

void fforward(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& E)
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

void backward(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& E, Eigen::SVectorXd& gradient, Eigen::SMatrixXd& Q)
{
    int l,m,p,n,nL=nbNeurons[L],jump;
    int N=globalIndices[2*L-1];

    Eigen::SVectorXd dzL(nL);
    Eigen::SVectorXd dz;
    Eigen::SMatrixXd dw;
    Eigen::SVectorXd Jpm(N);

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

void backwardJacob(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& J)
{
    int l,m,p,n,nL=nbNeurons[L],jump,nbLine=0;
    int N=globalIndices[2*L-1];

    Eigen::SVectorXd dzL(nL);
    Eigen::SVectorXd dz;
    Eigen::SMatrixXd dw;
    Eigen::SVectorXd Jpm(N);

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

void fforward_entropie(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::vector<Eigen::SMatrixXd>& As,
std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& E_inv, Eigen::SMatrixXd& E2_inv)
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

void backward_entropie(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& E_inv, Eigen::SMatrixXd& E2_inv, Eigen::SVectorXd& gradient, Eigen::SMatrixXd& Q)
{
    int l,m,p,n,nL=nbNeurons[L],jump;
    int N=globalIndices[2*L-1];

    Eigen::SVectorXd dzL(nL);
    Eigen::SVectorXd dz;
    Eigen::SMatrixXd dw;
    Eigen::SVectorXd Jpm(N);

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

void backwardJacob_entropie(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SMatrixXd& E_inv, Eigen::SMatrixXd& E2_inv, Eigen::SMatrixXd& J, Eigen::SMatrixXd& J2)
{
    int l,m,p,n,nL=nbNeurons[L],jump,nbLine=0;
    int N=globalIndices[2*L-1];

    Eigen::SVectorXd dzL(nL);
    Eigen::SVectorXd dz;
    Eigen::SMatrixXd dw;
    Eigen::SVectorXd Jpm(N);

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

Sdouble entropie(Eigen::SMatrixXd const& Y, Eigen::SMatrixXd const& outputs, int const& P, int const& nL)
{
    if(nL==1)
    {
        return (-1.0/Sdouble(P))*(Y.array()*outputs.array().log()+(1-Y.array())*(1-outputs.array()).log()).sum();
    }
    else
    {
        return (-1.0/Sdouble(P))*(Y.array()*outputs.array().log()).sum();
    }
}

void solve(Eigen::SVectorXd const& gradient, Eigen::SMatrixXd const& hessian, Eigen::SVectorXd& delta, std::string const method)
{

    if(method=="LLT"){Eigen::LLT<Eigen::SMatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="LDLT"){Eigen::LDLT<Eigen::SMatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient); }
    else if(method=="HouseholderQR"){Eigen::HouseholderQR<Eigen::SMatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="ColPivHouseholderQR"){Eigen::ColPivHouseholderQR<Eigen::SMatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="FullPivHouseholderQR"){Eigen::FullPivHouseholderQR<Eigen::SMatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="PartialPivLU"){Eigen::PartialPivLU<Eigen::SMatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="FullPivLU"){Eigen::FullPivLU<Eigen::SMatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="ConjugateGradient"){Eigen::ConjugateGradient<Eigen::SMatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="LeastSquaresConjugateGradient"){Eigen::LeastSquaresConjugateGradient<Eigen::SMatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="BiCGSTAB"){Eigen::BiCGSTAB<Eigen::SMatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}

}

void update(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Eigen::SVectorXd const& delta)
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
