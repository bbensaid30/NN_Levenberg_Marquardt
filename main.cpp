#include <iostream>
#include <string>
#include <iterator>
#include <vector>
#include <Eigen/Dense>
#include "data.h"
#include "init.h"
#include "training.h"
#include "test.h"

int main()
{
    std::vector<double> supParameters = {0,3};
    std::string distribution="normal"; int nbTirage=10000; double mu=10; double factor=10; double eps=std::pow(10,-7); int maxIter=2000; double epsNeight=std::pow(10,-7);
    testPolyThree(distribution,supParameters,nbTirage,mu,factor,eps,maxIter,epsNeight);

    //Load the data
//    std::vector<Eigen::MatrixXd> data(2);
//    int nbPoints=10;
//    data = sinc2(nbPoints);
//    double percTrain = 0.9;
//    std::vector<Eigen::MatrixXd> dataTrainTest(4);
//    dataTrainTest = trainTestData(data,percTrain);

//    std::vector<Eigen::MatrixXd> data(2);
//    Eigen::MatrixXd X(1,2), Y(1,2);
//    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
//    data[0]=X; data[1]=Y;
//
//     Problem variables
//    int const n0=data[0].rows(), nL=data[1].rows();
//    int N=0;
//    int const L=1;
//    std::vector<int> nbNeurons(L+1);
//    std::vector<int> globalIndices(2*L);
//    std::vector<std::string> activations(L);
//    std::vector<Eigen::MatrixXd> weights(L);
//    std::vector<Eigen::VectorXd> bias(L);
//
//    Architecture
//    int l;
//    nbNeurons[0]=n0;
//    nbNeurons[1]=nL;
//    activations[0]="polyThree";
//    for(l=0;l<L;l++)
//    {
//        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
//    }
//
//    Initialisation
//    std::vector<double> supParameters = {0,4};
//    initialisation(nbNeurons,weights,bias,supParameters,"normal");
//    std::cout << "La valeur initiale de w : " << weights[0] << std::endl;
//    std::cout << "La valeur initiale de b : " << bias[0] << std::endl;
//
//     Training
//    double mu=10, factor=10, eps=std::pow(10,-7);
//    int maxIter=200000;
//    std::map<std::string,double> study;
//    study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter);
//
//    std::cout << "La valeur finale de w : " << weights[0] << std::endl;
//    std::cout << "La valeur finale de b : " << bias[0] << std::endl;
//    std::cout << "Converge en " << (int)study["iter"] << " itérations" << std::endl;
//    std::cout << "La norme du gradient finale est : " << study["finalGradient"] << std::endl;
//    std::cout << "Le cout final: " << study["finalCost"] << std::endl;
//    std::cout << "La plus grande zone d'attraction commence à " << (int)study["startSequenceMax"] << " et termine à " << (int)study["endSequenceMax"] << std::endl;
//    std::cout << "La dernière zone d'attraction commence à " << (int)study["startSequenceFinal"] << std::endl;
//    std::cout << "Proportion de retours en arrière: " << study["propBack"] << std::endl;

//    std::vector<Eigen::MatrixXd> As(L+1); As[0]=dataTrainTest[2];
//    std::vector <Eigen::MatrixXd> slopes(L);
//    int const PTest = dataTrainTest[2].cols();
//    Eigen::MatrixXd ETest(nL,PTest);
//    fforward(dataTrainTest[2],dataTrainTest[3],L,PTest,nbNeurons,activations,weights,bias,As,slopes,ETest);
//    std::cout << "MSE = " << ETest.norm()/(PTest*nL) << std::endl;

    return 0;

}
