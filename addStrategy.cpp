#include "addStrategy.h"

bool addSimple(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
double const epsClose)
{
    int const taille = weightsList.size();
    int i=0;

    double dis;
    bool continuer=true;

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (dis>epsClose)
        {
            i++;
        }
        else{continuer=false;}
    }
    if(i==taille)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    else{return false;}

}


bool addCostCloser(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie, double const eRelative)
{
    int const taille = weightsList.size();
    int i=0, j=0, indiceCloser=0;

    std::vector<Eigen::MatrixXd> weightsLeft(L);
    std::vector<Eigen::VectorXd> biasLeft(L);
    std::vector<Eigen::MatrixXd> weightsRight(L);
    std::vector<Eigen::VectorXd> biasRight(L);
    std::vector<Eigen::MatrixXd> weightsInter(L);
    std::vector<Eigen::VectorXd> biasInter(L);
    std::vector<Eigen::MatrixXd> weightsMin(L);
    std::vector<Eigen::VectorXd> biasMin(L);
    double costInter, costMin;
    double gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    double dis, disMin;
    bool continuer=true;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient(N);
    Eigen::MatrixXd Q(N,N);

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if(i==0){disMin=dis;}
        if (dis>epsClose)
        {
            if (dis<disMin){disMin=dis; indiceCloser=i;}
            i++;
        }
        else{continuer=false;}
    }

    if(i==taille)
    {
        if(std::abs(currentCost-costs[indiceCloser])/currentCost>eRelative)
        {
            weightsList.push_back(weights); biasList.push_back(bias);
            costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
            return true;
        }
        std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
        std::copy(weightsList[indiceCloser].begin(),weightsList[indiceCloser].end(),weightsRight.begin()); std::copy(biasList[indiceCloser].begin(),biasList[indiceCloser].end(),biasRight.begin());
        gradientNormLeft=currentGradientNorm; gradientNormRight=gradientNorms[indiceCloser];
        if (gradientNormLeft<gradientNormRight)
        {
            gradientNormMin=gradientNormLeft;
            std::copy(weightsLeft.begin(),weightsLeft.end(),weightsMin.begin()); std::copy(biasLeft.begin(),biasLeft.end(),biasMin.begin());
        }
        else
        {
            gradientNormMin=gradientNormRight;
            std::copy(weightsRight.begin(),weightsRight.end(),weightsMin.begin()); std::copy(biasRight.begin(),biasRight.end(),biasMin.begin());
        }
        for(j=0;j<nbDichotomie;j++)
        {
            convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
            fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
            costInter=0.5*E.squaredNorm();
            gradient.setZero(); Q.setZero();
            backward(L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,E,gradient,Q);
            gradientNormInter=gradient.norm();
            if (std::abs(currentCost-costInter)/currentCost>eRelative)
            {
                weightsList.push_back(weights); biasList.push_back(bias);
                costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
                return true;
            }
            if(gradientNormLeft<gradientNormRight)
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (gradientNormInter<gradientNormMin)
            {
                costMin=costInter; gradientNormMin=gradientNormInter;
                std::copy(weightsInter.begin(),weightsInter.end(),weightsMin.begin()); std::copy(biasInter.begin(),biasInter.end(),biasMin.begin());
            }
        }

        std::copy(weightsMin.begin(),weightsMin.end(),weightsList[indiceCloser].begin()); std::copy(biasMin.begin(),biasMin.end(),biasList[indiceCloser].begin());
        costs[indiceCloser]=costMin; gradientNorms[indiceCloser]=gradientNormMin;
        return false;
    }
    else{return false;}
}

bool addCostAll(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie, double const eRelative)
{
    int const taille = weightsList.size();
    int i=0,j=0, k=0;

    std::vector<Eigen::MatrixXd> weightsLeft(L);
    std::vector<Eigen::VectorXd> biasLeft(L);
    std::vector<Eigen::MatrixXd> weightsRight(L);
    std::vector<Eigen::VectorXd> biasRight(L);
    std::vector<Eigen::MatrixXd> weightsInter(L);
    std::vector<Eigen::VectorXd> biasInter(L);
    std::vector<Eigen::MatrixXd> weightsMin(L);
    std::vector<Eigen::VectorXd> biasMin(L);
    double costLeft, costInter, costRight, costMin;
    double gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    double dis;
    bool continuer=true, plat=false;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient(N);
    Eigen::MatrixXd Q(N,N);

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (dis>epsClose)
        {
            i++;
        }
        else{continuer=false;}
    }

    if(i==taille)
    {
        while(k<taille && !plat)
        {
            if(std::abs(currentCost-costs[k])/currentCost>eRelative)
            {
                k++;
                continue;
            }
            std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
            std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
            costLeft=currentCost; costRight=costs[k];
            j=0;
            while(j<nbDichotomie)
            {
                convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
                fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
                costInter=0.5*E.squaredNorm();
                if (std::abs(currentCost-costInter)/currentCost>eRelative)
                {
                    k++;
                    break;
                }
                if(costLeft<costRight)
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }
                j++;
            }
            if(j==nbDichotomie)
            {
                plat=true;
            }

        }

    }
    else
    {
        return false;
    }

    if(plat)
    {
        std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
        std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
        costLeft=currentCost; costRight=costs[k];
        gradientNormLeft=currentGradientNorm; gradientNormRight=gradientNorms[k];
        if (gradientNormLeft<gradientNormRight)
        {
            gradientNormMin=gradientNormLeft; costMin=costLeft;
            std::copy(weightsLeft.begin(),weightsLeft.end(),weightsMin.begin()); std::copy(biasLeft.begin(),biasLeft.end(),biasMin.begin());
        }
        else
        {
            gradientNormMin=gradientNormRight; costMin=costRight;
            std::copy(weightsRight.begin(),weightsRight.end(),weightsMin.begin()); std::copy(biasRight.begin(),biasRight.end(),biasMin.begin());
        }
        for(j=0;j<nbDichotomie;j++)
        {
            convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
            fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
            costInter=0.5*E.squaredNorm();
            gradient.setZero(); Q.setZero();
            backward(L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,E,gradient,Q);
            gradientNormInter=gradient.norm();
            if(gradientNormLeft<gradientNormRight)
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (gradientNormInter<gradientNormMin)
            {
                costMin=costInter; gradientNormMin=gradientNormInter;
                std::copy(weightsInter.begin(),weightsInter.end(),weightsMin.begin()); std::copy(biasInter.begin(),biasInter.end(),biasMin.begin());
            }
        }

        std::copy(weightsMin.begin(),weightsMin.end(),weightsList[k].begin()); std::copy(biasMin.begin(),biasMin.end(),biasList[k].begin());
        costs[k]=costMin; gradientNorms[k]=gradientNormMin;
        return false;
    }
    else
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
}

bool addCostAllSet(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie, double const eRelative)
{
    int const taille = weightsList.size();
    int i=0,j=0;

    std::vector<Eigen::MatrixXd> weightsLeft(L);
    std::vector<Eigen::VectorXd> biasLeft(L);
    std::vector<Eigen::MatrixXd> weightsRight(L);
    std::vector<Eigen::VectorXd> biasRight(L);
    std::vector<Eigen::MatrixXd> weightsInter(L);
    std::vector<Eigen::VectorXd> biasInter(L);
    std::vector<Eigen::MatrixXd> weightsMin(L);
    std::vector<Eigen::VectorXd> biasMin(L);
    double costLeft, costInter, costRight, costMin;
    double gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    double dis;
    bool continuer=true;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient(N);
    Eigen::MatrixXd Q(N,N);

    std::vector<int> indicePlat;

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (dis>epsClose)
        {
            i++;
        }
        else{continuer=false;}
    }

    if(i==taille)
    {
        for(size_t k=0; k<weightsList.size(); k++)
        {
            if(std::abs(currentCost-costs[k])/currentCost>eRelative)
            {
                continue;
            }
            std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
            std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
            costLeft=currentCost; costRight=costs[k];
            j=0;
            while(j<nbDichotomie)
            {
                convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
                fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
                costInter=0.5*E.squaredNorm();
                if (std::abs(currentCost-costInter)/currentCost>eRelative)
                {
                    break;
                }
                if(costLeft<costRight)
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }
                j++;
            }
            if(j==nbDichotomie)
            {
                indicePlat.push_back(k);
            }

        }

    }
    else
    {
        return false;
    }

    if(indicePlat.size()==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    else
    {
        for(size_t k=0; k<indicePlat.size(); k++)
        {
            std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
            std::copy(weightsList[indicePlat[k]].begin(),weightsList[indicePlat[k]].end(),weightsRight.begin()); std::copy(biasList[indicePlat[k]].begin(),biasList[indicePlat[k]].end(),biasRight.begin());
            costLeft=currentCost; costRight=costs[indicePlat[k]];
            gradientNormLeft=currentGradientNorm; gradientNormRight=gradientNorms[indicePlat[k]];
            if (gradientNormLeft<gradientNormRight)
            {
                gradientNormMin=gradientNormLeft; costMin=costLeft;
                std::copy(weightsLeft.begin(),weightsLeft.end(),weightsMin.begin()); std::copy(biasLeft.begin(),biasLeft.end(),biasMin.begin());
            }
            else
            {
                gradientNormMin=gradientNormRight; costMin=costRight;
                std::copy(weightsRight.begin(),weightsRight.end(),weightsMin.begin()); std::copy(biasRight.begin(),biasRight.end(),biasMin.begin());
            }
            for(j=0;j<nbDichotomie;j++)
            {
                convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
                fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
                costInter=0.5*E.squaredNorm();
                gradient.setZero(); Q.setZero();
                backward(L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,E,gradient,Q);
                gradientNormInter=gradient.norm();
                if(gradientNormLeft<gradientNormRight)
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }

                if (gradientNormInter<gradientNormMin)
                {
                    costMin=costInter; gradientNormMin=gradientNormInter;
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsMin.begin()); std::copy(biasInter.begin(),biasInter.end(),biasMin.begin());
                }
            }

            std::copy(weightsMin.begin(),weightsMin.end(),weightsList[indicePlat[k]].begin()); std::copy(biasMin.begin(),biasMin.end(),biasList[indicePlat[k]].begin());
            costs[indicePlat[k]]=costMin; gradientNorms[indicePlat[k]]=gradientNormMin;
        }
        return false;

    }
}

bool addCostSd(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie, double const eRelative)
{
    int const taille = weightsList.size();
    int i=0, j, k=0;

    std::vector<Eigen::MatrixXd> weightsLeft(L);
    std::vector<Eigen::VectorXd> biasLeft(L);
    std::vector<Eigen::MatrixXd> weightsRight(L);
    std::vector<Eigen::VectorXd> biasRight(L);
    std::vector<Eigen::MatrixXd> weightsInter(L);
    std::vector<Eigen::VectorXd> biasInter(L);
    std::vector<Eigen::MatrixXd> weightsMin(L);
    std::vector<Eigen::VectorXd> biasMin(L);
    double costLeft, costInter, costRight, costMin;
    double gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    double dis;
    bool continuer=true, plat=false;

    double meanInter, sdInter;
    std::vector<double> costsRelative;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient(N);
    Eigen::MatrixXd Q(N,N);

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (dis>epsClose)
        {
            i++;
        }
        else{continuer=false;}
    }

    if(i==taille)
    {
        while(k<taille && !plat)
        {
            costsRelative.push_back(std::abs(currentCost-costs[k])/currentCost);
            std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
            std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
            costLeft=currentCost; costRight=costs[k];
            for(j=0;j<nbDichotomie;j++)
            {
                convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
                fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
                costInter=0.5*E.squaredNorm();
                if(costLeft<costRight)
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }
                costsRelative.push_back(std::abs(currentCost-costInter)/currentCost);
            }
            meanInter=mean(costsRelative);
            sdInter=sd(costsRelative,meanInter);
            costsRelative.clear();
            if(sdInter<eRelative){plat=true;}
            else{k++;}
        }

    }
    else
    {
        return false;
    }

    if(plat)
    {
        std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
        std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
        costLeft=currentCost; costRight=costs[k];
        gradientNormLeft=currentGradientNorm; gradientNormRight=gradientNorms[k];
        if (gradientNormLeft<gradientNormRight)
        {
            gradientNormMin=gradientNormLeft; costMin=costLeft;
            std::copy(weightsLeft.begin(),weightsLeft.end(),weightsMin.begin()); std::copy(biasLeft.begin(),biasLeft.end(),biasMin.begin());
        }
        else
        {
            gradientNormMin=gradientNormRight; costMin=costRight;
            std::copy(weightsRight.begin(),weightsRight.end(),weightsMin.begin()); std::copy(biasRight.begin(),biasRight.end(),biasMin.begin());
        }
        for(j=0;j<nbDichotomie;j++)
        {
            convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
            fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
            costInter=0.5*E.squaredNorm();
            gradient.setZero(); Q.setZero();
            backward(L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,E,gradient,Q);
            gradientNormInter=gradient.norm();
            if(gradientNormLeft<gradientNormRight)
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (gradientNormInter<gradientNormMin)
            {
                costMin=costInter; gradientNormMin=gradientNormInter;
                std::copy(weightsInter.begin(),weightsInter.end(),weightsMin.begin()); std::copy(biasInter.begin(),biasInter.end(),biasMin.begin());
            }
        }

        std::copy(weightsMin.begin(),weightsMin.end(),weightsList[k].begin()); std::copy(biasMin.begin(),biasMin.end(),biasList[k].begin());
        costs[k]=costMin; gradientNorms[k]=gradientNormMin;
        return false;
    }
    else
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
}

bool addCostSdAbs(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie, double const eAbs)
{
    int const taille = weightsList.size();
    int i=0, j, k=0;

    std::vector<Eigen::MatrixXd> weightsLeft(L);
    std::vector<Eigen::VectorXd> biasLeft(L);
    std::vector<Eigen::MatrixXd> weightsRight(L);
    std::vector<Eigen::VectorXd> biasRight(L);
    std::vector<Eigen::MatrixXd> weightsInter(L);
    std::vector<Eigen::VectorXd> biasInter(L);
    std::vector<Eigen::MatrixXd> weightsMin(L);
    std::vector<Eigen::VectorXd> biasMin(L);
    double costLeft, costInter, costRight, costMin;
    double gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    double dis;
    bool continuer=true, plat=false;

    double meanInter, sdInter;
    std::vector<double> costsAbs;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient(N);
    Eigen::MatrixXd Q(N,N);

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (dis>epsClose)
        {
            i++;
        }
        else{continuer=false;}
    }

    if(i==taille)
    {
        while(k<taille && !plat)
        {
            costsAbs.push_back(std::abs(costs[k]-currentCost));
            std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
            std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
            costLeft=currentCost; costRight=costs[k];
            for(j=0;j<nbDichotomie;j++)
            {
                convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
                fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
                costInter=0.5*E.squaredNorm();
                if(costLeft<costRight)
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }
                costsAbs.push_back(std::abs(costInter-currentCost));
            }
            meanInter=mean(costsAbs);
            sdInter=sd(costsAbs,meanInter);
            costsAbs.clear();
            if(sdInter<eAbs){plat=true;}
            else{k++;}
        }

    }
    else
    {
        return false;
    }

    if(plat)
    {
        std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
        std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
        costLeft=currentCost; costRight=costs[k];
        gradientNormLeft=currentGradientNorm; gradientNormRight=gradientNorms[k];
        if (gradientNormLeft<gradientNormRight)
        {
            gradientNormMin=gradientNormLeft; costMin=costLeft;
            std::copy(weightsLeft.begin(),weightsLeft.end(),weightsMin.begin()); std::copy(biasLeft.begin(),biasLeft.end(),biasMin.begin());
        }
        else
        {
            gradientNormMin=gradientNormRight; costMin=costRight;
            std::copy(weightsRight.begin(),weightsRight.end(),weightsMin.begin()); std::copy(biasRight.begin(),biasRight.end(),biasMin.begin());
        }
        for(j=0;j<nbDichotomie;j++)
        {
            convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
            fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
            costInter=0.5*E.squaredNorm();
            gradient.setZero(); Q.setZero();
            backward(L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,E,gradient,Q);
            gradientNormInter=gradient.norm();
            if(gradientNormLeft<gradientNormRight)
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (gradientNormInter<gradientNormMin)
            {
                costMin=costInter; gradientNormMin=gradientNormInter;
                std::copy(weightsInter.begin(),weightsInter.end(),weightsMin.begin()); std::copy(biasInter.begin(),biasInter.end(),biasMin.begin());
            }
        }

        std::copy(weightsMin.begin(),weightsMin.end(),weightsList[k].begin()); std::copy(biasMin.begin(),biasMin.end(),biasList[k].begin());
        costs[k]=costMin; gradientNorms[k]=gradientNormMin;
        return false;
    }
    else
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
}

bool addCostMedian(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie, double const eRelative)
{
    int const taille = weightsList.size();
    int i=0, j, k=0;

    std::vector<Eigen::MatrixXd> weightsLeft(L);
    std::vector<Eigen::VectorXd> biasLeft(L);
    std::vector<Eigen::MatrixXd> weightsRight(L);
    std::vector<Eigen::VectorXd> biasRight(L);
    std::vector<Eigen::MatrixXd> weightsInter(L);
    std::vector<Eigen::VectorXd> biasInter(L);
    std::vector<Eigen::MatrixXd> weightsMin(L);
    std::vector<Eigen::VectorXd> biasMin(L);
    double costLeft, costInter, costRight, costMin;
    double gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    double dis;
    bool continuer=true, plat=false;

    double medianInter;
    std::vector<double> costsRelative;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient(N);
    Eigen::MatrixXd Q(N,N);

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (dis>epsClose)
        {
            i++;
        }
        else{continuer=false;}
    }

    if(i==taille)
    {
        while(k<taille && !plat)
        {
            costsRelative.push_back(std::abs(currentCost-costs[k])/currentCost);
            std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
            std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
            costLeft=currentCost; costRight=costs[k];
            for(j=0;j<nbDichotomie;j++)
            {
                convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
                fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
                costInter=0.5*E.squaredNorm();
                if(costLeft<costRight)
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }
                costsRelative.push_back(std::abs(currentCost-costInter)/currentCost);
            }
            medianInter = median(costsRelative);
            costsRelative.clear();
            if(medianInter<eRelative){plat=true;}
            else{k++;}
        }

    }
    else
    {
        return false;
    }

    if(plat)
    {
        std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
        std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
        costLeft=currentCost; costRight=costs[k];
        gradientNormLeft=currentGradientNorm; gradientNormRight=gradientNorms[k];
        if (gradientNormLeft<gradientNormRight)
        {
            gradientNormMin=gradientNormLeft; costMin=costLeft;
            std::copy(weightsLeft.begin(),weightsLeft.end(),weightsMin.begin()); std::copy(biasLeft.begin(),biasLeft.end(),biasMin.begin());
        }
        else
        {
            gradientNormMin=gradientNormRight; costMin=costRight;
            std::copy(weightsRight.begin(),weightsRight.end(),weightsMin.begin()); std::copy(biasRight.begin(),biasRight.end(),biasMin.begin());
        }
        for(j=0;j<nbDichotomie;j++)
        {
            convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
            fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
            costInter=0.5*E.squaredNorm();
            gradient.setZero(); Q.setZero();
            backward(L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,E,gradient,Q);
            gradientNormInter=gradient.norm();
            if(gradientNormLeft<gradientNormRight)
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (gradientNormInter<gradientNormMin)
            {
                costMin=costInter; gradientNormMin=gradientNormInter;
                std::copy(weightsInter.begin(),weightsInter.end(),weightsMin.begin()); std::copy(biasInter.begin(),biasInter.end(),biasMin.begin());
            }
        }

        std::copy(weightsMin.begin(),weightsMin.end(),weightsList[k].begin()); std::copy(biasMin.begin(),biasMin.end(),biasList[k].begin());
        costs[k]=costMin; gradientNorms[k]=gradientNormMin;
        return false;
    }
    else
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
}

bool addCostSdNonEquivalent(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie, double const eRelative)
{
    int const taille = weightsList.size();
    int i=0, j, k=0;

    std::vector<Eigen::MatrixXd> weightsLeft(L);
    std::vector<Eigen::VectorXd> biasLeft(L);
    std::vector<Eigen::MatrixXd> weightsRight(L);
    std::vector<Eigen::VectorXd> biasRight(L);
    std::vector<Eigen::MatrixXd> weightsInter(L);
    std::vector<Eigen::VectorXd> biasInter(L);
    std::vector<Eigen::MatrixXd> weightsMin(L);
    std::vector<Eigen::VectorXd> biasMin(L);
    double costLeft, costInter, costRight, costMin;
    double gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    double dis;
    bool continuer=true, plat=false;

    double meanInter, sdInter;
    std::vector<double> costsRelative;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient(N);
    Eigen::MatrixXd Q(N,N);

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (dis>epsClose)
        {
            i++;
        }
        else{continuer=false;}
    }

    if(i==taille)
    {
        while(k<taille && !plat)
        {
            costsRelative.push_back(std::abs(currentCost-costs[k])/currentCost);
            std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
            std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
            costLeft=currentCost; costRight=costs[k];
            for(j=0;j<nbDichotomie;j++)
            {
                convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
                fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
                costInter=0.5*E.squaredNorm();
                if(costLeft<costRight)
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }
                costsRelative.push_back(std::abs(currentCost-costInter)/currentCost);
            }
            meanInter=mean(costsRelative);
            sdInter=sd(costsRelative,meanInter);
            costsRelative.clear();
            if(sdInter<eRelative){plat=true;}
            else{k++;}
        }

    }
    else
    {
        return false;
    }

    if(plat)
    {
        std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
        std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
        costLeft=currentCost; costRight=costs[k];
        gradientNormLeft=currentGradientNorm; gradientNormRight=gradientNorms[k];
        if (gradientNormLeft<gradientNormRight)
        {
            gradientNormMin=gradientNormLeft; costMin=costLeft;
            std::copy(weightsLeft.begin(),weightsLeft.end(),weightsMin.begin()); std::copy(biasLeft.begin(),biasLeft.end(),biasMin.begin());
        }
        else
        {
            gradientNormMin=gradientNormRight; costMin=costRight;
            std::copy(weightsRight.begin(),weightsRight.end(),weightsMin.begin()); std::copy(biasRight.begin(),biasRight.end(),biasMin.begin());
        }
        for(j=0;j<nbDichotomie;j++)
        {
            convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
            fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
            costInter=0.5*E.squaredNorm();
            gradient.setZero(); Q.setZero();
            backward(L,P,nbNeurons,globalIndices,weightsInter,biasInter,As,slopes,E,gradient,Q);
            gradientNormInter=gradient.norm();
            if(gradientNormLeft<gradientNormRight)
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (gradientNormInter<gradientNormMin)
            {
                costMin=costInter; gradientNormMin=gradientNormInter;
                std::copy(weightsInter.begin(),weightsInter.end(),weightsMin.begin()); std::copy(biasInter.begin(),biasInter.end(),biasMin.begin());
            }
        }

        std::copy(weightsMin.begin(),weightsMin.end(),weightsList[k].begin()); std::copy(biasMin.begin(),biasMin.end(),biasList[k].begin());
        costs[k]=costMin; gradientNorms[k]=gradientNormMin;
        return false;
    }
    else
    {
        i=0;
        while (i<taille && continuer)
        {
            dis = std::abs(currentCost-costs[i]);
            if (dis>epsClose)
            {
                i++;
            }
            else{continuer=false;}
        }
        if(i==taille)
        {
            weightsList.push_back(weights); biasList.push_back(bias);
            costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
            return true;
        }
        else{return false;}
    }
}

bool addPoint(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::vector<std::vector<Eigen::MatrixXd>>& weightsList,
std::vector<std::vector<Eigen::VectorXd>>& biasList, double const& currentCost, double const& currentGradientNorm, std::vector<double>& costs, std::vector<double>& gradientNorms,
Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
double const epsClose, int const nbDichotomie, double const flat, std::string const strategy)
{
    if (strategy=="CostCloser")
    {
        return addCostCloser(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,X,Y,L,P,nbNeurons,globalIndices,activations,epsClose,nbDichotomie,flat);
    }
    else if(strategy=="CostAll")
    {
        return addCostAll(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,X,Y,L,P,nbNeurons,globalIndices,activations,epsClose,nbDichotomie,flat);
    }
    else if(strategy=="CostAllSet")
    {
        return addCostAllSet(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,X,Y,L,P,nbNeurons,globalIndices,activations,epsClose,nbDichotomie,flat);
    }
    else if(strategy=="CostSd")
    {
        return addCostSd(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,X,Y,L,P,nbNeurons,globalIndices,activations,epsClose,nbDichotomie,flat);
    }
    else if(strategy=="CostSdAbs")
    {
        return addCostSdAbs(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,X,Y,L,P,nbNeurons,globalIndices,activations,epsClose,nbDichotomie,flat);
    }
    else if(strategy=="CostMedian")
    {
        return addCostMedian(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,X,Y,L,P,nbNeurons,globalIndices,activations,epsClose,nbDichotomie,flat);
    }
    else if(strategy=="CostSdNonEquivalent")
    {
        return addCostSdNonEquivalent(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,X,Y,L,P,nbNeurons,globalIndices,activations,epsClose,nbDichotomie,flat);
    }
    else
    {
        return addSimple(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,epsClose);
    }

}
