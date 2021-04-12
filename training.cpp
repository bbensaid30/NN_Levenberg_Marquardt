#include "training.h"

void propagation(Eigen::MatrixXd& A, Eigen::MatrixXd const& Y, int const L, int const nbNeurons[], int const P, std::string const activations[], Eigen::MatrixXd const weights[],
Eigen::VectorXd const bias[], Eigen::MatrixXd As[], Eigen::MatrixXd slopes[], Eigen::MatrixXd E, Eigen::VectorXd dz, Eigen::VectorXd Jpm, Eigen::VectorXd gradient, Eigen::MatrixXd Q)
{
    int l,m,p,n,nL=nbNeurons[L];
    int N=0, jump=0, indice;

    //forward
    As[0]=A;
    for (l=0;l<L;l++)
    {
        A = weights[l]*A;
        A.colwise() += bias[l];
        activation(activations[l], A, slopes[l]);
        As[l+1]=A;
        N+=(nbNeurons[l]+1)*nbNeurons[l+1];
    }
    E=Y-A;

    //backward
    for (p=0;p<P;p++)
    {
        for (m=0;m<nL;m++)
        {
            for (n=0;n<nL;n++)
            {
                dz[n] = (n==m) ? -slopes[L](m,p) : 0;
            }
            indice=N;
            for (l=L;l>0;l--)
            {
                indice-=jump;
                jump=nbNeurons[l+1]*nbNeurons[l];
                Jpm.segment(indice,-jump)=(dz*As[l-1].col(p).transpose()).conservativeResize(jump,1);
                indice-=jump;
                jump+=nbNeurons[l+1];
                Jpm.segment(indice,-jump)=dz;

                dz=(weights[l].transpose()*dz).cwiseProduct(slopes[l-1].col(p));
            }
        }
    }
}
