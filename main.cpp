#include <iostream>
#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include <random>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"
#include <EigenRand/EigenRand>

#include "testLM.h"
#include "testSGD.h"
#include "propagation.h"

#include "init.h"
#include "data.h"
#include "SGDs.h"
#include "study_base.h"
#include "utilities.h"
#include "perte.h"

int main()
{
    std::string const distribution="uniform";
    std::vector<double> const supParameters={-3,3};
    int const nbTirage=10000;
    std::string const algo="RMSProp";
    Sdouble const learning_rate=0.1;
    Sdouble const beta1=0.9;
    Sdouble const beta2=0.999;
    int const batch_size=2;
    Sdouble const eps=std::pow(10,-7);
    int const maxIter=20000;
    Sdouble const epsNeight=std::pow(10,-3);


    testSGD_PolyTwo(distribution,supParameters,nbTirage,algo,learning_rate,batch_size,beta1,beta2,eps,maxIter,epsNeight);

    return 0;
}
