#include "data.h"

std::vector<Eigen::SMatrixXd> sineWave(int const& nbPoints)
{
    std::vector<Eigen::SMatrixXd> data(2);
    Eigen::SMatrixXd X(1,nbPoints);
    Eigen::SMatrixXd Y(1,nbPoints);
    X.row(0) = Eigen::SArrayXd::LinSpaced(nbPoints,-1,1);

    data[0] = X;
    data[1] = 0.5+0.25*(3*M_PI*X.array()).sin();

    return data;
}

std::vector<Eigen::SMatrixXd> squareWave(int const& nbPoints, Sdouble const frequence)
{
    std::vector<Eigen::SMatrixXd> data(2);
    Eigen::SMatrixXd X(1,nbPoints);
    X.row(0) = Eigen::SArrayXd::LinSpaced(nbPoints,-1,1);

    data[0] = X;
    data[1] = 2*(2*(frequence*X.array()).floor()-(2*frequence*X.array()).floor())+1;

    return data;
}

std::vector<Eigen::SMatrixXd> sinc1(int const& nbPoints)
{
    std::vector<Eigen::SMatrixXd> data(2);
    Eigen::SMatrixXd X(1,nbPoints);
    Eigen::SMatrixXd Y(1,nbPoints);
    X.row(0) = Eigen::SArrayXd::LinSpaced(nbPoints,-1,1);

    for (int i=0;i<nbPoints;i++)
    {
        if(X(0,i)<std::pow(10,-16))
        {
            Y(0,i)=1;
        }
        else
        {
            Y(0,i)=Sstd::sin(X(0,i))/X(0,i);
        }
    }

    data[0]=X;
    data[1]=Y;

    return data;
}


std::vector<Eigen::SMatrixXd> sinc2(int const& nbPoints)
{
    std::vector<Eigen::SMatrixXd> data(2);
    Eigen::SMatrixXd X(2,nbPoints*nbPoints);
    Eigen::SMatrixXd Y(1,nbPoints*nbPoints);
    Eigen::SArrayXd points = Eigen::SArrayXd::LinSpaced(nbPoints,-1,1);
    int i,j=0;
    Sdouble norme;

    while (j<nbPoints*nbPoints)
    {
        for (i=0;i<nbPoints;i++)
        {
            X(0,j) = points[j/nbPoints];
            X(1,j) = points[i];
            norme = Sstd::sqrt(Sstd::pow(X(0,j),2)+Sstd::pow(X(1,j),2));
            if (norme < std::pow(10,-16)){Y(0,j) = 1;}
            else {Y(0,j) = Sstd::sin(norme)/norme;}
            j++;
        }
    }

    data[0] = X;
    data[1] = Y;

    return data;
}

std::vector<Eigen::SMatrixXd> exp2(int const& nbPoints)
{
    std::vector<Eigen::SMatrixXd> data(2);
    Eigen::SMatrixXd X(2,nbPoints*nbPoints);
    Eigen::SMatrixXd Y(1,nbPoints*nbPoints);
    Eigen::SArrayXd points = Eigen::SArrayXd::LinSpaced(nbPoints,0,4);
    int i,j=0;
    Sdouble norme;

    while (j<nbPoints*nbPoints)
    {
        for (i=0;i<nbPoints;i++)
        {
            X(0,j) = points[j/nbPoints];
            X(1,j) = points[i];
            Y(0,j) = 4*Sstd::exp(-0.15*Sstd::pow((X(0,j)-4),2)-0.5*Sstd::pow(X(1,j)-3,2));
            j++;
        }
    }

    data[0] = X;
    data[1] = Y;

    return data;
}

std::vector<Eigen::SMatrixXd> twoSpiral(int const& nbPoints)
{
    std::vector<Eigen::SMatrixXd> data(2);
    Eigen::SMatrixXd X(2,2*nbPoints);
    Eigen::SMatrixXd Y(1,2*nbPoints);

    Eigen::SArrayXd theta = Eigen::SArrayXd::LinSpaced(nbPoints,0,2*M_PI);
    Eigen::SArrayXd r0 = 2*theta+M_PI;
    int i;
    Eigen::Rand::Vmt19937_64 gen{ 100 };

    for(i=0;i<nbPoints;i++)
    {
        X(0,i) = Sstd::cos(theta[i])*r0[i];
        X(1,i) = Sstd::sin(theta[i])*r0[i];
        Y(0,i)=0;
    }
    for(i=nbPoints;i<2*nbPoints;i++)
    {
        X(0,i) = -Sstd::cos(theta[i-nbPoints])*r0[i-nbPoints];
        X(1,i) = -Sstd::sin(theta[i-nbPoints])*r0[i-nbPoints];
        Y(0,i)=1;
    }

    data[0]=X; data[1]=Y;
    return data;

}

std::vector<Eigen::SMatrixXd> MNIST(std::string const& nameFileTrain, std::string const& nameFileTest)
{
    std::ifstream fileTrain(("Data/"+nameFileTrain).c_str());
    std::ifstream fileTest(("Data/"+nameFileTest).c_str());

    Eigen::SMatrixXd XTrain(784,60000), XTest(784,10000);
    Eigen::SMatrixXd YTrain = Eigen::SMatrixXd::Zero(10,60000), YTest = Eigen::SMatrixXd::Zero(10,10000);
    std::vector<Eigen::SMatrixXd> data(4);

    std::string line, subChaine;
    std::stringstream ss(line);
    int i=0,j=0;

   if(fileTrain)
   {

      while(getline(fileTrain, line))
      {
            ss=(std::stringstream)line;
            std::getline(ss, subChaine, ','); YTrain(i,strtol(subChaine.c_str(),NULL,10))=1;
            while (std::getline(ss, subChaine, ','))
            {
                XTrain(i,j)=Sdouble(strtol(subChaine.c_str(),NULL,10));
                j++;
            }
            i++; j=0;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier d'entraînement en lecture." << std::endl;
   }

   i=0;j=0;

   if(fileTest)
   {

      while(getline(fileTest, line))
      {
            ss=(std::stringstream)line;
            getline(ss, subChaine, ','); YTest(i,strtol(subChaine.c_str(),NULL,10))=1;
            while (std::getline(ss, subChaine, ','))
            {
                XTest(i,j)=Sdouble(strtol(subChaine.c_str(),NULL,10));
                j++;
            }
            i++; j=0;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier de test en lecture." << std::endl;
   }

   data[0]=XTrain; data[1]=YTrain; data[2]=XTest; data[3]=YTest;

   return data;

}

std::vector<Eigen::SMatrixXd> trainTestData(std::vector <Eigen::SMatrixXd> const& data, Sdouble const& percTrain, bool const reproductible)
{
    std::vector<Eigen::SMatrixXd> dataTrainTest(4);
    int const P = data[0].cols(), n0 = data[0].rows(), nL=data[1].rows();
    int const sizeTrain = (int) (percTrain*P);
    int const sizeTest = P-sizeTrain;
    std::vector<char> nbChosen(P);
    Eigen::SMatrixXd XTrain(n0,sizeTrain), XTest(n0,sizeTest), YTrain(nL,sizeTrain), YTest(nL,sizeTest);
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
        iter++;
       }while(nbChosen[randomNumber]=='y' && iter<10*P);
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
