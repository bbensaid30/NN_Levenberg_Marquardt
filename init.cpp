#include "init.h"

void simple(std::vector<int> const nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias)
{
    int const L = nbNeurons.size()-1;
    int l;

    for(l=0;l<L;l++)
    {
        weights[l]=Eigen::MatrixXd::Random(nbNeurons[l+1],nbNeurons[l]);
        bias[l]=Eigen::VectorXd::Zero(nbNeurons[l+1]);
    }
}

void uniform(std::vector<int> const nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const& a, double const& b)
{
    std::random_device rd;
    std::mt19937 gen(rd());  //here you could also set a seed
    std::uniform_real_distribution<double> distrib(a, b);

    int const L = nbNeurons.size()-1;
    int l;

    for (l=0;l<L;l++)
    {
        Eigen::MatrixXd random_matrix = Eigen::MatrixXd::Zero(nbNeurons[l+1],nbNeurons[l]).unaryExpr([&](double dummy){return distrib(gen);});
        Eigen::VectorXd random_vector = Eigen::VectorXd::Zero(nbNeurons[l+1]).unaryExpr([&](float dummy){return distrib(gen);});


        weights[l] = random_matrix;
        bias[l] = random_vector;
    }
}

void normal(std::vector<int> const nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const& mu, double const& sigma)
{
    std::random_device rd;
    std::mt19937 gen(rd());  //here you could also set a seed
    std::normal_distribution<double> distrib(mu, sigma);

    int const L = nbNeurons.size()-1;
    int l;

    for (l=0;l<L;l++)
    {
        Eigen::MatrixXd random_matrix = Eigen::MatrixXd::Zero(nbNeurons[l+1],nbNeurons[l]).unaryExpr([&](double dummy){return distrib(gen);});
        Eigen::VectorXd random_vector = Eigen::VectorXd::Zero(nbNeurons[l+1]).unaryExpr([&](float dummy){return distrib(gen);});

        weights[l] = random_matrix;
        bias[l] = random_vector;
    }
}

void initialisation(std::vector<int> const nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<double> const& supParameters, std::string generator)
{
    if (generator=="uniform")
    {
        uniform(nbNeurons,weights,bias,supParameters[0],supParameters[1]);
    }
    else if (generator=="normal")
    {
        normal(nbNeurons,weights,bias,supParameters[0],supParameters[1]);
    }
    else
    {
        simple(nbNeurons,weights,bias);
    }
}
