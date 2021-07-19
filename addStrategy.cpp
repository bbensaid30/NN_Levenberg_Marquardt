#include "addStrategy.h"

bool addSimple(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Sdouble const& epsClose)
{
    int const taille = weightsList.size();
    int i=0;

    Sdouble dis;
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
        if (dis-std::abs(dis.error)>epsClose)
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


bool addCostCloser(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eRelative)
{
    int const taille = weightsList.size();
    int i=0, j=0, indiceCloser=0;

    std::vector<Eigen::SMatrixXd> weightsLeft(L);
    std::vector<Eigen::SVectorXd> biasLeft(L);
    std::vector<Eigen::SMatrixXd> weightsRight(L);
    std::vector<Eigen::SVectorXd> biasRight(L);
    std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    std::vector<Eigen::SMatrixXd> weightsMin(L);
    std::vector<Eigen::SVectorXd> biasMin(L);
    Sdouble costInter, costMin;
    Sdouble gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    Sdouble dis, disMin;
    bool continuer=true;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient(N);
    Eigen::SMatrixXd Q(N,N);

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
        if (!std::signbit((dis-epsClose).number))
        {
            if (std::signbit((dis-disMin).number)){disMin=dis; indiceCloser=i;}
            i++;
        }
        else{continuer=false;}
    }

    Sdouble errorRelative;
    if(i==taille)
    {
        errorRelative = Sstd::abs(currentCost-costs[indiceCloser])/currentCost;
        if(!std::signbit((errorRelative-eRelative).number))
        {
            weightsList.push_back(weights); biasList.push_back(bias);
            costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
            return true;
        }
        std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
        std::copy(weightsList[indiceCloser].begin(),weightsList[indiceCloser].end(),weightsRight.begin()); std::copy(biasList[indiceCloser].begin(),biasList[indiceCloser].end(),biasRight.begin());
        gradientNormLeft=currentGradientNorm; gradientNormRight=gradientNorms[indiceCloser];
        if (std::signbit((gradientNormLeft-gradientNormRight).number))
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
            errorRelative = Sstd::abs(currentCost-costInter)/currentCost;
            if (!std::signbit((errorRelative-eRelative).number))
            {
                weightsList.push_back(weights); biasList.push_back(bias);
                costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
                return true;
            }
            if(std::signbit((gradientNormLeft-gradientNormRight).number))
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (std::signbit((gradientNormInter-gradientNormMin).number))
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

bool addCostAll(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eRelative)
{
    int const taille = weightsList.size();
    int i=0,j=0, k=0;

    std::vector<Eigen::SMatrixXd> weightsLeft(L);
    std::vector<Eigen::SVectorXd> biasLeft(L);
    std::vector<Eigen::SMatrixXd> weightsRight(L);
    std::vector<Eigen::SVectorXd> biasRight(L);
    std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    std::vector<Eigen::SMatrixXd> weightsMin(L);
    std::vector<Eigen::SVectorXd> biasMin(L);
    Sdouble costLeft, costInter, costRight, costMin;
    Sdouble gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    Sdouble dis;
    bool continuer=true, plat=false;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient(N);
    Eigen::SMatrixXd Q(N,N);

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (!std::signbit((dis-epsClose).number))
        {
            i++;
        }
        else{continuer=false;}
    }

    Sdouble errorRelative;
    if(i==taille)
    {
        while(k<taille && !plat)
        {
            errorRelative = Sstd::abs(currentCost-costs[k])/currentCost;
            if(!std::signbit((errorRelative-eRelative).number))
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
                errorRelative = Sstd::abs(currentCost-costInter)/currentCost;
                if (!std::signbit((errorRelative-eRelative).number))
                {
                    k++;
                    break;
                }
                if(std::signbit((costLeft-costRight).number))
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
        if (std::signbit((gradientNormLeft-gradientNormRight).number))
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
            if(std::signbit((gradientNormLeft-gradientNormRight).number))
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (std::signbit((gradientNormInter-gradientNormMin).number))
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

bool addCostAllSet(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eRelative)
{
    int const taille = weightsList.size();
    int i=0,j=0;

    std::vector<Eigen::SMatrixXd> weightsLeft(L);
    std::vector<Eigen::SVectorXd> biasLeft(L);
    std::vector<Eigen::SMatrixXd> weightsRight(L);
    std::vector<Eigen::SVectorXd> biasRight(L);
    std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    std::vector<Eigen::SMatrixXd> weightsMin(L);
    std::vector<Eigen::SVectorXd> biasMin(L);
    Sdouble costLeft, costInter, costRight, costMin;
    Sdouble gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    Sdouble dis;
    bool continuer=true;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient(N);
    Eigen::SMatrixXd Q(N,N);

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
        if (!std::signbit((dis-epsClose).number))
        {
            i++;
        }
        else{continuer=false;}
    }

    Sdouble errorRelative;
    if(i==taille)
    {
        for(size_t k=0; k<weightsList.size(); k++)
        {
            errorRelative = Sstd::abs(currentCost-costs[k])/currentCost;
            if(!std::signbit((errorRelative-eRelative).number))
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
                errorRelative = Sstd::abs(currentCost-costInter)/currentCost;
                if (!std::signbit((errorRelative-eRelative).number))
                {
                    break;
                }
                if(std::signbit((costLeft-costRight).number))
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
            if (std::signbit((gradientNormLeft-gradientNormRight).number))
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
                if(std::signbit((gradientNormLeft-gradientNormRight).number))
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }

                if (std::signbit((gradientNormInter-gradientNormMin).number))
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

bool addCostSd(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eRelative)
{
    int const taille = weightsList.size();
    int i=0, j, k=0;

    std::vector<Eigen::SMatrixXd> weightsLeft(L);
    std::vector<Eigen::SVectorXd> biasLeft(L);
    std::vector<Eigen::SMatrixXd> weightsRight(L);
    std::vector<Eigen::SVectorXd> biasRight(L);
    std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    std::vector<Eigen::SMatrixXd> weightsMin(L);
    std::vector<Eigen::SVectorXd> biasMin(L);
    Sdouble costLeft, costInter, costRight, costMin;
    Sdouble gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    Sdouble dis;
    bool continuer=true, plat=false;

    Sdouble meanInter, sdInter;
    std::vector<Sdouble> costsRelative;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient(N);
    Eigen::SMatrixXd Q(N,N);

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (!std::signbit((dis-epsClose).number))
        {
            i++;
        }
        else{continuer=false;}
    }

    if(i==taille)
    {
        while(k<taille && !plat)
        {
            costsRelative.push_back(Sstd::abs(currentCost-costs[k])/currentCost);
            std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
            std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
            costLeft=currentCost; costRight=costs[k];
            for(j=0;j<nbDichotomie;j++)
            {
                convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
                fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
                costInter=0.5*E.squaredNorm();
                if(std::signbit((costLeft-costRight).number))
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }
                costsRelative.push_back(Sstd::abs(currentCost-costInter)/currentCost);
            }
            meanInter=mean(costsRelative);
            sdInter=sd(costsRelative,meanInter);
            costsRelative.clear();
            if(std::signbit((sdInter-eRelative).number)){plat=true;}
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
        if (std::signbit((gradientNormLeft-gradientNormRight).number))
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
            if(std::signbit((gradientNormLeft-gradientNormRight).number))
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (std::signbit((gradientNormInter-gradientNormMin).number))
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

bool addCostSdAbs(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eAbs)
{
    int const taille = weightsList.size();
    int i=0, j, k=0;

    std::vector<Eigen::SMatrixXd> weightsLeft(L);
    std::vector<Eigen::SVectorXd> biasLeft(L);
    std::vector<Eigen::SMatrixXd> weightsRight(L);
    std::vector<Eigen::SVectorXd> biasRight(L);
    std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    std::vector<Eigen::SMatrixXd> weightsMin(L);
    std::vector<Eigen::SVectorXd> biasMin(L);
    Sdouble costLeft, costInter, costRight, costMin;
    Sdouble gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    Sdouble dis;
    bool continuer=true, plat=false;

    Sdouble meanInter, sdInter;
    std::vector<Sdouble> costsAbs;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient(N);
    Eigen::SMatrixXd Q(N,N);

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (!std::signbit((dis-epsClose).number))
        {
            i++;
        }
        else{continuer=false;}
    }

    if(i==taille)
    {
        while(k<taille && !plat)
        {
            costsAbs.push_back(Sstd::abs(costs[k]-currentCost));
            std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
            std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
            costLeft=currentCost; costRight=costs[k];
            for(j=0;j<nbDichotomie;j++)
            {
                convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
                fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
                costInter=0.5*E.squaredNorm();
                if(std::signbit((costLeft-costRight).number))
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }
                costsAbs.push_back(Sstd::abs(costInter-currentCost));
            }
            meanInter=mean(costsAbs);
            sdInter=sd(costsAbs,meanInter);
            costsAbs.clear();
            if(std::signbit((sdInter-eAbs).number)){plat=true;}
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
        if (std::signbit((gradientNormLeft-gradientNormRight).number))
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
            if(std::signbit((gradientNormLeft-gradientNormRight).number))
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (std::signbit((gradientNormInter-gradientNormMin).number))
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

bool addCostMedian(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& eRelative)
{
    int const taille = weightsList.size();
    int i=0, j, k=0;

    std::vector<Eigen::SMatrixXd> weightsLeft(L);
    std::vector<Eigen::SVectorXd> biasLeft(L);
    std::vector<Eigen::SMatrixXd> weightsRight(L);
    std::vector<Eigen::SVectorXd> biasRight(L);
    std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    std::vector<Eigen::SMatrixXd> weightsMin(L);
    std::vector<Eigen::SVectorXd> biasMin(L);
    Sdouble costLeft, costInter, costRight, costMin;
    Sdouble gradientNormLeft, gradientNormInter, gradientNormRight, gradientNormMin;
    Sdouble dis;
    bool continuer=true, plat=false;

    Sdouble medianInter;
    std::vector<Sdouble> costsRelative;

    int N=globalIndices[2*L-1];
    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient(N);
    Eigen::SMatrixXd Q(N,N);

    if (taille==0)
    {
        weightsList.push_back(weights); biasList.push_back(bias);
        costs.push_back(currentCost); gradientNorms.push_back(currentGradientNorm);
        return true;
    }
    while (i<taille && continuer)
    {
        dis=distance(weights,bias,weightsList[i],biasList[i],"2");
        if (!std::signbit((dis-epsClose).number))
        {
            i++;
        }
        else{continuer=false;}
    }

    if(i==taille)
    {
        while(k<taille && !plat)
        {
            costsRelative.push_back(Sstd::abs(currentCost-costs[k])/currentCost);
            std::copy(weights.begin(),weights.end(),weightsLeft.begin()); std::copy(bias.begin(),bias.end(),biasLeft.begin());
            std::copy(weightsList[k].begin(),weightsList[k].end(),weightsRight.begin()); std::copy(biasList[k].begin(),biasList[k].end(),biasRight.begin());
            costLeft=currentCost; costRight=costs[k];
            for(j=0;j<nbDichotomie;j++)
            {
                convexCombination(weightsLeft,biasLeft,weightsRight,biasRight,weightsInter,biasInter,L,0.5);
                fforward(X,Y,L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes,E);
                costInter=0.5*E.squaredNorm();
                if(std::signbit((costLeft-costRight).number))
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
                }
                else
                {
                    std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
                }
                costsRelative.push_back(Sstd::abs(currentCost-costInter)/currentCost);
            }
            medianInter = median(costsRelative);
            costsRelative.clear();
            if(std::signbit((medianInter-eRelative).number)){plat=true;}
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
        if (std::signbit((gradientNormLeft-gradientNormRight).number))
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
            if(std::signbit((gradientNormLeft-gradientNormRight).number))
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsRight.begin()); std::copy(biasInter.begin(),biasInter.end(),biasRight.begin());
            }
            else
            {
                std::copy(weightsInter.begin(),weightsInter.end(),weightsLeft.begin()); std::copy(biasInter.begin(),biasInter.end(),biasLeft.begin());
            }

            if (std::signbit((gradientNormInter-gradientNormMin).number))
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




bool addPoint(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::vector<std::vector<Eigen::SMatrixXd>>& weightsList,
std::vector<std::vector<Eigen::SVectorXd>>& biasList, Sdouble const& currentCost, Sdouble const& currentGradientNorm, std::vector<Sdouble>& costs, std::vector<Sdouble>& gradientNorms,
Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, Sdouble const& epsClose, int const& nbDichotomie, Sdouble const& flat, std::string const& strategy)
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
    else
    {
        return addSimple(weights,bias,weightsList,biasList,currentCost,currentGradientNorm,costs,gradientNorms,epsClose);
    }

}
