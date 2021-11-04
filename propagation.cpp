#include "propagation.h"

//------------------------------------------------------------------ Propagation directe ----------------------------------------------------------------------------------------

void fforward(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes)
{
    int l;
    for (l=0;l<L;l++)
    {
        As[l+1] = weights[l]*As[l];
        As[l+1].colwise() += bias[l];
        activation(activations[l], As[l+1], slopes[l]);
    }
}

Sdouble risk(Eigen::SMatrixXd const& Y, int const& P, Eigen::SMatrixXd const& output_network, std::string const& type_perte)
{
    Sdouble cost=0;
    for(int p=0; p<P; p++)
    {
        cost+=L(Y.col(p),output_network.col(p),type_perte);
    }
    return cost;
}

//------------------------------------------------------------------ Rétropropagation ---------------------------------------------------------------------------------------------

void backward(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SVectorXd& gradient, std::string const& type_perte)
{
    int l,p,n,nL=nbNeurons[L],jump;
    int N=globalIndices[2*L-1];

    Eigen::SMatrixXd L_derivative(nL,P);
    Eigen::SVectorXd LP(nL);

    Eigen::SMatrixXd dzL(nL,P);
    Eigen::SMatrixXd dz;
    Eigen::SMatrixXd dw;
    Eigen::SVectorXd db;

    for(p=0; p<P; p++)
    {
        FO_L(Y.col(p),As[L].col(p),LP,type_perte);
        L_derivative.col(p)=LP;
    }

    dzL = L_derivative.cwiseProduct(slopes[L-1]);

    dz=dzL;
    jump=nbNeurons[L]*nbNeurons[L-1];
    dw = dz*(As[L-1].transpose());
    db = dz.rowwise().sum();
    dw.resize(jump,1);
    gradient.segment(globalIndices[2*(L-1)]-jump,jump)=dw;
    jump=nbNeurons[L];
    gradient.segment(globalIndices[2*(L-1)+1]-jump,jump)=db;
    for (l=L-1;l>0;l--)
    {
        dz=(weights[l].transpose()*dz).cwiseProduct(slopes[l-1]);

        jump=nbNeurons[l]*nbNeurons[l-1];
        dw=dz*(As[l-1].transpose());
        db = dz.rowwise().sum();
        dw.resize(jump,1);
        gradient.segment(globalIndices[2*(l-1)]-jump,jump)=dw;
        jump=nbNeurons[l];
        gradient.segment(globalIndices[2*(l-1)+1]-jump,jump)=db;
    }
}


//Cas où L'' diagonale (la plupart du temps)
void QSO_backward(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
std::vector<Eigen::SMatrixXd>& As, std::vector<Eigen::SMatrixXd>& slopes, Eigen::SVectorXd& gradient, Eigen::SMatrixXd& Q, std::string const& type_perte)
{
    int l,m,p,n,nL=nbNeurons[L],jump;
    int N=globalIndices[2*L-1];

    Eigen::SVectorXd LP(nL);
    Eigen::SMatrixXd LPP(nL,nL);
    Eigen::SVectorXd dzL(nL);
    Eigen::SVectorXd dz;
    Eigen::SMatrixXd dw;
    Eigen::SVectorXd Jpm(N);

    for (p=0;p<P;p++)
    {
        SO_L(Y.col(p),As[L].col(p),LP,LPP,type_perte);
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
            Q+=LPP(m,m)*Jpm*Jpm.transpose();
            gradient+=-LP(m)*Jpm;
        }
    }

}

void QSO_backwardJacob(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
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
