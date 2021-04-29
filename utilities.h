#ifndef UTILITIES
#define UTILITIES

#include <vector>
#include <map>
#include <Eigen/Dense>


int proportion(Eigen::VectorXd const& currentPoint, std::vector<Eigen::VectorXd> const& points, std::vector<double>& proportions,  std::vector<double>& distances, double const& epsNeight);

#endif
