#ifndef STUDY_GRAPHICS
#define STUDY_GRAPHICS

#include <vector>
#include <iostream>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"
#include "matplotlibcpp.h"

#include "study_base.h"

void nbMinsFlats(std::vector<Eigen::SMatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& algo, Sdouble const& epsClose, int const& nbDichotomie,
Sdouble const& eps, std::string const& folder, std::string const fileExtension, Sdouble const& x, Sdouble const& a, Sdouble const& b, Sdouble const& pas,
std::string const& strategy);


#endif
