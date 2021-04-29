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
