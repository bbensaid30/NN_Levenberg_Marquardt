#include "data.h"

std::vector<Eigen::MatrixXd> sineWave(int nbPoints)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(1,nbPoints);
    X.row(0) = Eigen::ArrayXd::LinSpaced(nbPoints,-1,1);

    data[0] = X;
    data[1] = 0.5+0.25*(3*M_PI*X.array()).sin();

    return data;
}

std::vector<Eigen::MatrixXd> trainTestData(std::vector <Eigen::MatrixXd> data, double percTrain, bool reproductible)
{
    std::vector<Eigen::MatrixXd> dataTrainTest(4);
    int const P = data[0].cols(), n0 = data[0].rows(), nL=data[1].rows();
    int const sizeTrain = (int) (percTrain*P);
    int const sizeTest = P-sizeTrain;
    char nbChosen[P];
    Eigen::MatrixXd XTrain(n0,sizeTrain), XTest(n0,sizeTest), YTrain(nL,sizeTrain), YTest(nL,sizeTest);
    int i, j=0, iter=0, randomNumber;
    std::default_random_engine re;
    std::default_random_engine reAlea(time(0));
    std::uniform_int_distribution<int> distrib{0,P-1};

    for(i=0;i<P;i++) {nbChosen[i]='n';}

    if (reproductible){randomNumber=distrib(re);}
    else {randomNumber=distrib(reAlea);}
    nbChosen[randomNumber]='y';
    XTrain.col(0) = data[0].col(randomNumber);
    YTrain.col(0) = data[1].col(randomNumber);
    for (i=1;i<sizeTrain;i++)
    {
       do
       {
        if (reproductible){randomNumber=distrib(re);}
        else {randomNumber=distrib(reAlea);}
       }while(nbChosen[randomNumber]=='y');
       nbChosen[randomNumber]='y';

       XTrain.col(i) = data[0].col(randomNumber);
       YTrain.col(i) = data[1].col(randomNumber);
    }
    for (i=0;i<P;i++)
    {
        if (nbChosen[i]=='n')
        {
            XTest.col(j) = data[0].col(i);
            YTest.col(j) = data[1].col(i);
            j++;
        }
    }

    dataTrainTest[0] = XTrain; dataTrainTest[1] = YTrain; dataTrainTest[2] = XTest; dataTrainTest[3] = YTest;

    return dataTrainTest;
}
