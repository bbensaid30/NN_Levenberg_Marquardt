#include "init.h"

void simple(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed)
{
    std::random_device rd;
    Eigen::Rand::Vmt19937_64 gen{ seed };

    int const L = nbNeurons.size()-1;
    int l;

    for(l=0;l<L;l++)
    {
        weights[l] = convert(Eigen::Rand::balanced<Eigen::MatrixXd>(nbNeurons[l+1],nbNeurons[l],gen));
        bias[l] = convert(Eigen::Rand::balanced<Eigen::MatrixXd>(nbNeurons[l+1],1,gen));
    }
}

void uniform(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, double const& a, double const& b, unsigned const seed)
{

    std::random_device rd;
    Eigen::Rand::Vmt19937_64 gen{ seed };

    int const L = nbNeurons.size()-1;
    int l;

    for (l=0;l<L;l++)
    {
        weights[l] = convert((b-a)*Eigen::Rand::uniformReal<Eigen::MatrixXd>(nbNeurons[l+1],nbNeurons[l],gen).array()+a);
        bias[l] = convert((b-a)*Eigen::Rand::uniformReal<Eigen::MatrixXd>(nbNeurons[l+1],1,gen).array()+a);
    }
}

void normal(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, double const& mu, double const& sigma, unsigned const seed)
{
    std::random_device rd;
    Eigen::Rand::Vmt19937_64 gen{ seed };

    int const L = nbNeurons.size()-1;
    int l;

    for (l=0;l<L;l++)
    {
        weights[l] = convert(Eigen::Rand::normal<Eigen::MatrixXd>(nbNeurons[l+1],nbNeurons[l],gen,mu,sigma));
        bias[l] = convert(Eigen::Rand::normal<Eigen::MatrixXd>(nbNeurons[l+1],1,gen,mu,sigma));
    }
}

void initialisation(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::vector<double> const& supParameters,
std::string const& generator, unsigned const seed)
{
    if (generator=="uniform")
    {
        uniform(nbNeurons,weights,bias,supParameters[0],supParameters[1], seed);
    }
    else if (generator=="normal")
    {
        normal(nbNeurons,weights,bias,supParameters[0],supParameters[1], seed);
    }
    else
    {
        simple(nbNeurons,weights,bias);
    }
}
