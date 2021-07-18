#include "study_graphics.h"
namespace plt = matplotlibcpp;

void nbMinsFlats(std::vector<Eigen::SMatrixXd> const&  data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& algo, Sdouble const& epsClose, int const& nbDichotomie,
Sdouble const& eps, std::string const& folder, std::string const& fileExtension, Sdouble const& x, Sdouble const& a, Sdouble const& b, Sdouble const& pas,
std::string const& strategy)
{
    std::vector<double> flats;
    Sdouble flat;
    std::vector<int> nbsMins;
    int nbMin;
    Sdouble power=a;

    int const PTrain = data[0].cols();
    std::ostringstream epsStream;
    epsStream << eps;
    std::string epsString = epsStream.str();
    std::ostringstream PStream;
    PStream << PTrain;
    std::string PString = PStream.str();
    std::ofstream nbsMinsFlux(("Record/"+folder+"/nbsMins_"+algo+"_"+fileExtension+"(eps="+epsString+", P="+PString+").csv").c_str());

    while(power<b)
    {
        flat=Sstd::pow(x,power);
        flats.push_back((double)flat);
        nbMin = denombrementMinsPost(data,L,nbNeurons,globalIndices,activations,algo,epsClose,nbDichotomie,eps,10000,strategy,flat,folder,fileExtension);
        nbsMins.push_back(nbMin);
        nbsMinsFlux << flat << nbMin << std::endl;
        power+=pas;
    }

    plt::plot(flats,nbsMins,"bo-");
    plt::xlabel("flat threshold");
    plt::ylabel("Number of mins");
    plt::title("Number of mins as a function of flat threshold for strategy="+strategy);
    plt::show();

    std::cout << "Médiane du nb de mins: " << median(nbsMins) << std::endl;
    Sdouble moy = mean(nbsMins);
    std::cout << "Moyenne du nb de mins: " << moy << " avec écart-type: " << sd(nbsMins,moy) << std::endl;
    std::cout << "Min de nb de mins : " << minVector(nbsMins) << std::endl;

}

