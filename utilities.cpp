#include "utilities.h"

int proportion(Eigen::SVectorXd const& currentPoint, std::vector<Eigen::SVectorXd> const& points, std::vector<Sdouble>& proportions, std::vector<Sdouble>& distances, Sdouble const& epsNeight)
{
    int const nbPoints=points.size(), nbProportions=proportions.size();
    assert(nbPoints==nbProportions);

    int i=0;
    Sdouble distance;
    for(i=0;i<nbPoints;i++)
    {
        distance=(currentPoint-points[i]).norm();
        if (distance+std::abs(distance.error)<epsNeight)
        {
            proportions[i]++;
            distances[i]+=distance;
            return i;
        }
    }
    return -1;
}

Sdouble mean(std::vector<Sdouble> const& values)
{
    Sdouble sum = accumul(values);
    Sdouble mean = sum / values.size();

    return mean;
}
Sdouble mean(std::vector<int> const& values)
{
    int sum = std::accumulate(values.begin(), values.end(), 0.0);
    Sdouble mean = sum / values.size();

    return mean;
}

Sdouble sd(std::vector<Sdouble> const& values, Sdouble const& moy)
{
    Sdouble sq_sum = InnerProduct(values,values);
    Sdouble stdev = Sstd::sqrt(sq_sum / values.size() - Sstd::pow(moy,2));

    return stdev;
}
Sdouble sd(std::vector<int> const& values, Sdouble const& moy)
{
    int sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    Sdouble stdev = Sstd::sqrt(sq_sum / values.size() - Sstd::pow(moy,2));

    return stdev;
}

Sdouble median(std::vector<Sdouble>& values)
{
        int const taille = values.size();

        if (taille == 0)
                throw std::domain_error("median of an empty vector");

        std::sort(values.begin(), values.end());

        int const mid = taille/2;

        return taille % 2 == 0 ? (values[mid] + values[mid-1]) / 2.0 : values[mid];
}
int median(std::vector<int>& values)
{
        int const taille = values.size();

        if (taille == 0)
                throw std::domain_error("median of an empty vector");

        std::sort(values.begin(), values.end());

        int const mid = taille/2;

        return taille % 2 == 0 ? (values[mid] + values[mid-1]) / 2.0 : values[mid];
}

Sdouble minVector(std::vector<Sdouble> const& values)
{
    int taille = values.size();
    if (taille==0){throw std::domain_error("minimum of an empty vector");}

    Sdouble minimum = values[0];
    for (int i=0; i<taille ; i++)
    {
        if(values[i]<minimum){minimum=values[i];}
    }

    return minimum;
}
int minVector(std::vector<int> const& values)
{
    int taille = values.size();
    if (taille==0){throw std::domain_error("minimum of an empty vector");}

    int minimum = values[0];
    for (int i=0; i<taille ; i++)
    {
        if(values[i]<minimum){minimum=values[i];}
    }

    return minimum;
}


Sdouble distance(std::vector<Eigen::SMatrixXd> const& weightsPrec, std::vector<Eigen::SVectorXd> const& biasPrec,
std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::string const norm)
{
    Sdouble dis=0.0, disCurrent;
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
        return Sstd::sqrt(dis);
    }

}

Sdouble cosVector(Eigen::SVectorXd const& v1, Eigen::SVectorXd const& v2)
{
    Sdouble result;
    result = v1.dot(v2);
    return result/=v1.norm()*v2.norm();
}

void convexCombination(std::vector<Eigen::SMatrixXd> const& weights1, std::vector<Eigen::SVectorXd> const& bias1, std::vector<Eigen::SMatrixXd> const& weights2,
std::vector<Eigen::SVectorXd> const& bias2, std::vector<Eigen::SMatrixXd>& weightsInter, std::vector<Eigen::SVectorXd>& biasInter, int const& L, Sdouble const lambda)
{
    for(int l=0;l<L;l++)
    {
        weightsInter[l]=lambda*weights1[l]+(1-lambda)*weights2[l];
        biasInter[l]=lambda*bias1[l]+(1-lambda)*bias2[l];
    }
}

void nesterovCombination(std::vector<Eigen::SMatrixXd> const& weights1, std::vector<Eigen::SVectorXd> const& bias1, std::vector<Eigen::SMatrixXd> const& weights2,
std::vector<Eigen::SVectorXd> const& bias2, std::vector<Eigen::SMatrixXd>& weightsInter, std::vector<Eigen::SVectorXd>& biasInter, int const& L, Sdouble const& lambda)
{
    for(int l=0;l<L;l++)
    {
        weightsInter[l]= weights1[l]+lambda*(weights1[l]-weights2[l]);
        biasInter[l] = bias1[l]+lambda*(bias1[l]-bias2[l]);
    }
}

void tabToVector(std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd> const& bias, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
Eigen::SVectorXd& point)
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

void standardization(Eigen::SMatrixXd& X)
{
    int const dim=X.rows(), P=X.cols();
    Eigen::SVectorXd mean(dim), standardDeviation(dim);

    mean = X.rowwise().mean();
    for(int i=0; i<dim;i++)
    {
        X.array().row(i) -= mean(i);
    }

    standardDeviation = X.rowwise().squaredNorm()/(Sdouble(P));
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

void readMatrix(std::ifstream& flux, Eigen::SMatrixXd& result, int const& nbRows, int const& nbCols)
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

void readVector(std::ifstream& flux, Eigen::SVectorXd& result, int const& nbRows)
{
    std::string line;

    for(int i=0; i<nbRows; i++)
    {
        std::getline(flux, line);

        std::stringstream stream(line);
        stream >> result(i);
    }

}

Sdouble indexProperValues(Eigen::SMatrixXd const& H)
{
    Sdouble prop=0;
    Eigen::SelfAdjointEigenSolver<Eigen::SMatrixXd> eigensolver(H);
    Eigen::SVectorXd eivals = eigensolver.eigenvalues();

    int const taille = eivals.rows();
    for(int i=0; i<taille; i++)
    {
        if(eivals(i)<0){prop++;}
    }
    return prop/taille;
}


Sdouble expInv(Sdouble const& x)
{
    Sdouble const eps = std::pow(10,-14);
    if(Sstd::abs(x)<eps)
    {
        return 0;
    }
    else
    {
        return Sstd::exp(-1/(x*x));
    }
}
