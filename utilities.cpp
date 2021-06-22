#include "utilities.h"

int proportion(Eigen::VectorXd const& currentPoint, std::vector<Eigen::VectorXd> const& points, std::vector<double>& proportions, std::vector<double>& distances, double const& epsNeight)
{
    int const nbPoints=points.size(), nbProportions=proportions.size();
    assert(nbPoints==nbProportions);

    int i=0;
    double distance;
    for(i=0;i<nbPoints;i++)
    {
        distance=(currentPoint-points[i]).norm();
        if (distance<epsNeight)
        {
            proportions[i]++;
            distances[i]+=distance;
            return i;
        }
    }
    return -1;
}

double mean(std::vector<double> const& values)
{
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();

    return mean;
}

double sd(std::vector<double> const& values, double const& moy)
{
    double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / values.size() - std::pow(moy,2));

    return stdev;
}

double median(std::vector<double>& values)
{
        int const taille = values.size();

        if (taille == 0)
                throw std::domain_error("median of an empty vector");

        std::sort(values.begin(), values.end());

        int const mid = taille/2;

        return taille % 2 == 0 ? (values[mid] + values[mid-1]) / 2.0 : values[mid];
}


double distance(std::vector<Eigen::MatrixXd> const& weightsPrec, std::vector<Eigen::VectorXd> const& biasPrec,
std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::string norm)
{
    double dis=0.0, disCurrent;
    int const L = weights.size();
    int l;

    if(norm=="infinity")
    {
        for(l=0;l<L;l++)
        {
            disCurrent = (weights[l]-weightsPrec[l]).lpNorm<Eigen::Infinity>();
            if(disCurrent>dis){dis=disCurrent;}
            disCurrent = (bias[l]-biasPrec[l]).lpNorm<Eigen::Infinity>();
            if(disCurrent>dis){dis=disCurrent;}
        }
        return dis;
    }
    else
    {
        for(l=0;l<L;l++)
        {
            dis += (weights[l]-weightsPrec[l]).squaredNorm();
            dis += (bias[l]-biasPrec[l]).squaredNorm();
        }
        return std::sqrt(dis);
    }

}

double cosVector(Eigen::VectorXd const& v1, Eigen::VectorXd const& v2)
{
    double result;
    result = v1.dot(v2);
    return result/=v1.norm()*v2.norm();
}

void convexCombination(std::vector<Eigen::MatrixXd> const& weights1, std::vector<Eigen::VectorXd> const& bias1, std::vector<Eigen::MatrixXd> const& weights2,
std::vector<Eigen::VectorXd> const& bias2, std::vector<Eigen::MatrixXd>& weightsInter, std::vector<Eigen::VectorXd>& biasInter, int const& L, double const lambda)
{
    for(int l=0;l<L;l++)
    {
        weightsInter[l]=lambda*weights1[l]+(1-lambda)*weights2[l];
        biasInter[l]=lambda*bias1[l]+(1-lambda)*bias2[l];
    }
}

void tabToVector(std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd> const& bias, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
Eigen::VectorXd& point)
{
    int l, jump;

    for (l=0;l<L;l++)
    {
        jump=nbNeurons[l]*nbNeurons[l+1];
        weights[l].resize(jump,1);
        point.segment(globalIndices[2*l]-jump,jump)=weights[l];
        jump=nbNeurons[l+1];
        point.segment(globalIndices[2*l+1]-jump,jump)=bias[l];

        weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
    }
}

void standardization(Eigen::MatrixXd& X)
{
    int const dim=X.rows(), P=X.cols();
    Eigen::VectorXd mean(dim), standardDeviation(dim);

    mean = X.rowwise().mean();
    for(int i=0; i<dim;i++)
    {
        X.array().row(i) -= mean(i);
    }

    standardDeviation = X.rowwise().squaredNorm()/((double)(P));
    standardDeviation.cwiseSqrt();

    for(int i=0; i<dim;i++)
    {
        X.row(i) /= standardDeviation(i);
    }
}

int nbLines(std::ifstream& flux) {
    std::string s;

    unsigned int nb = 0;
    while(std::getline(flux,s)) {++nb;}

    return nb;

}

void readMatrix(std::ifstream& flux, Eigen::MatrixXd& result, int const& nbRows, int const& nbCols)
{
    int cols, rows;
    std::string line;

    for(rows=0; rows<nbRows; rows++)
    {
        std::getline(flux, line);

        std::stringstream stream(line);
        cols=0;
        while(! stream.eof())
        {
            stream >> result(rows,cols);
            cols++;
        }
    }

}

void readVector(std::ifstream& flux, Eigen::VectorXd& result, int const& nbRows)
{
    std::string line;

    for(int i=0; i<nbRows; i++)
    {
        std::getline(flux, line);

        std::stringstream stream(line);
        stream >> result(i);

    }

}
