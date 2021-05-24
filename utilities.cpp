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

void tabToVector(std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd> const& bias, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
Eigen::VectorXd& point)
{
    int l, jump;

    jump=nbNeurons[L]*nbNeurons[L-1];
    weights[L].resize(jump,1);
    point.segment(globalIndices[2*(L-1)]-jump,jump)=weights[L-1];
    jump=nbNeurons[L];
    point.segment(globalIndices[2*(L-1)+1]-jump,jump)=bias[L-1];
    for (l=L-1;l>0;l--)
    {
        jump=nbNeurons[l]*nbNeurons[l-1];
        weights[l-1].resize(jump,1);
        point.segment(globalIndices[2*(l-1)]-jump,jump)=weights[l-1];
        jump=nbNeurons[l];
        point.segment(globalIndices[2*(l-1)+1]-jump,jump)=bias[l-1];

        weights[l-1].resize(nbNeurons[l],nbNeurons[l-1]);
    }
}

bool testPoint(Eigen::VectorXd const& point, std::vector<Eigen::VectorXd>& points, double const epsClose)
{
    int const taille = points.size();
    int i=0;

    if (taille==0)
    {
        points.push_back(point);
        return true;
    }

    while ((points[i]-point).norm()>epsClose){i++;}
    if(i==taille)
    {
        points.push_back(point);
        return true;
    }
    else{return false;}
}




