#include "SGDs.h"

std::map<std::string,Sdouble> SGD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_SGD_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    int number_data = batch_size, indice_begin;

    Eigen::SMatrixXd echantillonX, echantillonY;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble moyGradientNorm = 1000, sommeGradient;
    Sdouble prop_entropie=0, prop_initial_ineq=0;
    Sdouble costInit, cost, costPrec;

    while (moyGradientNorm+std::abs(moyGradientNorm.error)>eps && iter<maxIter)
    {
        std::shuffle(indices.data(), indices.data()+P, eng);
        X = X*indices.asPermutation();
        Y = Y*indices.asPermutation();

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        sommeGradient=0;
        for(batch=0; batch<number_batch;batch++)
        {
            indice_begin = batch*batch_size;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
                if(tracking){cost = risk(Y,P,As[L],type_perte); costInit=cost;}
            }

            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();

            if(tracking)
            {
                costPrec = cost;
                cost = risk(Y,P,As[L],type_perte);
                if(!std::signbit((cost-costPrec).number)){prop_entropie++;}
                if(!std::signbit((cost-costInit).number)){prop_initial_ineq++;}
            }

        }

        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }

            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();
        }


        moyGradientNorm = sommeGradient/number_batch;
        if(numericalNoise(moyGradientNorm)){break;}

        iter++;
    }

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;
    if(tracking){study["prop_entropie"] = prop_entropie/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> SGD_Ito(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& eps, int const& maxIter, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_SGD_Ito"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    int number_data = batch_size, indice_begin;


    Eigen::SMatrixXd echantillonX, echantillonY;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble moyGradientNorm = 1000, sommeGradient;


    while (moyGradientNorm+std::abs(moyGradientNorm.error)>eps && iter<maxIter)
    {
        std::shuffle(indices.data(), indices.data()+P, eng);
        X = X*indices.asPermutation();
        Y = Y*indices.asPermutation();

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        sommeGradient=0;
        for(batch=0; batch<number_batch;batch++)
        {
            indice_begin = batch*batch_size;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }

            if (batch==0){update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);}
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();

        }
        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }

            if (batch==0){update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);}
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();
        }


        moyGradientNorm = sommeGradient/number_batch;
        if(numericalNoise(moyGradientNorm)){break;}

        iter++;
    }

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    Sdouble cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;

    return study;

}


std::map<std::string,Sdouble> SGD_clipping(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
Sdouble const& clip, int const& batch_size, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_SGD_clipping_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    int number_data = batch_size, indice_begin;

    Eigen::SMatrixXd echantillonX, echantillonY;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble moyGradientNorm = 1000, sommeGradient, h;
    Sdouble prop_entropie=0, prop_initial_ineq=0;
    Sdouble costInit, cost, costPrec;

    while (moyGradientNorm+std::abs(moyGradientNorm.error)>eps && iter<maxIter)
    {
        std::shuffle(indices.data(), indices.data()+P, eng);
        X = X*indices.asPermutation();
        Y = Y*indices.asPermutation();

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        sommeGradient=0;
        for(batch=0; batch<number_batch;batch++)
        {
            indice_begin = batch*batch_size;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
                h = Sstd::min(learning_rate,(clip*learning_rate)/gradient.norm());
                if(tracking){cost = risk(Y,P,As[L],type_perte); costInit=cost;}
            }

            update(L,nbNeurons,globalIndices,weights,bias,-h*gradient);
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            h = Sstd::min(learning_rate,(clip*learning_rate)/gradient.norm());
            sommeGradient += gradient.norm();

            if(tracking)
            {
                costPrec = cost;
                cost = risk(Y,P,As[L],type_perte);
                if(std::signbit((cost-costPrec).number)){prop_entropie++;}
                if(std::signbit((cost-costInit).number)){prop_initial_ineq++;}
            }

        }

        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
                h = Sstd::min(learning_rate,(clip*learning_rate)/gradient.norm());
            }

            update(L,nbNeurons,globalIndices,weights,bias,-h*gradient);
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            h = Sstd::min(learning_rate,(clip*learning_rate)/gradient.norm());
            sommeGradient += gradient.norm();
        }


        moyGradientNorm = sommeGradient/number_batch;
        if(numericalNoise(moyGradientNorm)){break;}

        iter++;
    }

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;
    if(tracking){study["prop_entropie"] = prop_entropie/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> Momentum_Euler(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_Euler_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    int number_data = batch_size, indice_begin;
    Sdouble beta_bar = beta1/learning_rate;


    Eigen::SMatrixXd echantillonX, echantillonY;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);

    Sdouble moyGradientNorm = 1000, sommeGradient;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    Sdouble EmPrec,Em,cost,costInit;

    while (moyGradientNorm+std::abs(moyGradientNorm.error)>eps && iter<maxIter)
    {
        std::shuffle(indices.data(), indices.data()+P, eng);
        X = X*indices.asPermutation();
        Y = Y*indices.asPermutation();

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        sommeGradient=0;
        for(batch=0; batch<number_batch;batch++)
        {
            indice_begin = batch*batch_size;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

                if(tracking)
                {
                    cost = risk(echantillonY,number_data,As[L],type_perte); costInit = cost;
                    Em = beta_bar*cost;
                }
            }
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1);
            moment1 = (1-beta1)*moment1 + beta1*gradient;
            if(track_continuous)
            {
                condition = moment1.dot(gradient);
                if(condition>=0){continuous_entropie++;}
            }


            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();

            if(tracking)
            {
                cost = risk(echantillonY,number_data,As[L],type_perte);
                EmPrec = Em; Em = 0.5*moment1.squaredNorm()+beta_bar*cost;
                if(Em-EmPrec>0)
                {
                    Em_count+=1;
                }
                if(cost-costInit>0)
                {
                    prop_initial_ineq+=1;
                }
            }
        }

        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1);
            moment1 = (1-beta1)*moment1 + beta1*gradient;

            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();
        }


        moyGradientNorm = sommeGradient/number_batch;
        if(numericalNoise(moyGradientNorm)){break;}

        iter++;
    }

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> Momentum(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    int number_data = batch_size, indice_begin;
    Sdouble beta_bar = beta1/learning_rate;


    Eigen::SMatrixXd echantillonX, echantillonY;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);

    Sdouble moyGradientNorm = 1000, sommeGradient;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    Sdouble EmPrec,Em,cost,costInit;

    while (moyGradientNorm+std::abs(moyGradientNorm.error)>eps && iter<maxIter)
    {
        std::shuffle(indices.data(), indices.data()+P, eng);
        X = X*indices.asPermutation();
        Y = Y*indices.asPermutation();

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        sommeGradient=0;
        for(batch=0; batch<number_batch;batch++)
        {
            indice_begin = batch*batch_size;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

                if(tracking)
                {
                    cost = risk(echantillonY,number_data,As[L],type_perte); costInit = cost;
                    Em = beta_bar*cost;
                }
            }
            moment1 = (1-beta1)*moment1 + beta1*gradient;
            if(track_continuous)
            {
                condition = moment1.dot(gradient);
                if(condition>=0){continuous_entropie++;}
            }

            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1);

            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();

            if(tracking)
            {
                cost = risk(echantillonY,number_data,As[L],type_perte);
                EmPrec = Em; Em = 0.5*moment1.squaredNorm()+beta_bar*cost;
                if(Em-EmPrec>0)
                {
                    Em_count+=1;
                }
                if(cost-costInit>0)
                {
                    prop_initial_ineq+=1;
                }
            }
        }

        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }
            moment1 = (1-beta1)*moment1 + beta1*gradient;

            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1);
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();
        }


        moyGradientNorm = sommeGradient/number_batch;
        if(numericalNoise(moyGradientNorm)){break;}

        iter++;
    }

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> AdaGrad(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& eps, int const& maxIter, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_AdaGrad_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    int number_data = batch_size, indice_begin;


    Eigen::SMatrixXd echantillonX, echantillonY;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SArrayXd moment = Eigen::SArrayXd::Zero(N);

    Sdouble moyGradientNorm = 1000, sommeGradient;


    while (moyGradientNorm+std::abs(moyGradientNorm.error)>eps && iter<maxIter)
    {
        std::shuffle(indices.data(), indices.data()+P, eng);
        X = X*indices.asPermutation();
        Y = Y*indices.asPermutation();

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        sommeGradient=0;
        for(batch=0; batch<number_batch;batch++)
        {
            indice_begin = batch*batch_size;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }

            moment += gradient.array().pow(2);
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient.array()*((moment+std::pow(10,-10)).rsqrt()));
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();
        }
        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }

            moment += gradient.array().pow(2);
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient.array()*((moment+std::pow(10,-10)).rsqrt()));
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();
        }


        moyGradientNorm = sommeGradient/number_batch;
        if(numericalNoise(moyGradientNorm)){break;}

        iter++;
    }

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    Sdouble cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;

    return study;

}

std::map<std::string,Sdouble> RMSProp(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta2, Sdouble const& eps, int const& maxIter, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_RMSProp_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    int number_data = batch_size, indice_begin;


    Eigen::SMatrixXd echantillonX, echantillonY;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SArrayXd moment2 = Eigen::SArrayXd::Zero(N);

    Sdouble moyGradientNorm = 1000, sommeGradient;


    while (moyGradientNorm+std::abs(moyGradientNorm.error)>eps && iter<maxIter)
    {
        std::shuffle(indices.data(), indices.data()+P, eng);
        X = X*indices.asPermutation();
        Y = Y*indices.asPermutation();

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        sommeGradient=0;
        for(batch=0; batch<number_batch;batch++)
        {
            indice_begin = batch*batch_size;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }
            moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);

            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient.array()*((moment2+std::pow(10,-10)).rsqrt()));
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            sommeGradient += gradient.squaredNorm();
        }
        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }
            moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);

            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient.array()*((moment2+std::pow(10,-10)).rsqrt()));
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            sommeGradient += gradient.norm();
        }


        moyGradientNorm = sommeGradient/number_batch;
        if(numericalNoise(moyGradientNorm)){break;}

        iter++;
    }

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    Sdouble cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;

    return study;

}

std::map<std::string,Sdouble> Adam(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream gradientNormFlux(("Record/gradientNorm_Adam_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_Adam_"+fileExtension+".csv").c_str());
    if(!costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    int number_data = batch_size, indice_begin;


    Eigen::SMatrixXd echantillonX, echantillonY;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);
    Eigen::SArrayXd moment2 = Eigen::SArrayXd::Zero(N);

    Sdouble moyGradientNorm = 1000, sommeGradient;
    Sdouble cost;

    while (moyGradientNorm+std::abs(moyGradientNorm.error)>eps && iter<maxIter)
    {
        std::shuffle(indices.data(), indices.data()+P, eng);
        X = X*indices.asPermutation();
        Y = Y*indices.asPermutation();

        sommeGradient=0;
        for(batch=0; batch<number_batch;batch++)
        {
            indice_begin = batch*batch_size;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
                if(tracking){cost=risk(Y,P,As[L],type_perte);}
                if(record)
                {
                    if(tracking){costsFlux << cost.number << std::endl;}
                    gradientNormFlux << gradient.norm().number << std::endl;
                }
            }

            moment1 = (1-beta1)*moment1 + beta1*gradient;
            moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1.array()*((moment2+std::pow(10,-10)).rsqrt()));
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            sommeGradient += gradient.norm();

        }

        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }

            moment1 = (1-beta1)*moment1 + beta1*gradient;
            moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1.array()*((moment2+std::pow(10,-10)).rsqrt()));
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            sommeGradient += gradient.norm();
        }


        moyGradientNorm = sommeGradient/number_batch;
        if(numericalNoise(moyGradientNorm)){break;}

        iter++;

        if(tracking){cost = risk(Y,P,As[L],type_perte);}
        if(record)
        {
            if(tracking){costsFlux << cost.number << std::endl;}
            gradientNormFlux << moyGradientNorm.number << std::endl;
        }
    }

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;

    return study;

}

std::map<std::string,Sdouble> AMSGrad(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_AMSGrad_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    int number_data = batch_size, indice_begin;


    Eigen::SMatrixXd echantillonX, echantillonY;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);
    Eigen::SArrayXd moment2 = Eigen::SArrayXd::Zero(N);
    Eigen::SArrayXd max_moment2;

    Sdouble moyGradientNorm = 1000, sommeGradient;


    while (moyGradientNorm+std::abs(moyGradientNorm.error)>eps && iter<maxIter)
    {
        std::shuffle(indices.data(), indices.data()+P, eng);
        X = X*indices.asPermutation();
        Y = Y*indices.asPermutation();

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        sommeGradient=0;
        for(batch=0; batch<number_batch;batch++)
        {
            indice_begin = batch*batch_size;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }

            moment1 = (1-beta1)*moment1 + beta1*gradient;
            moment2 = (1-beta2)*moment2 + beta2*gradient.array().pow(2);
            if(iter==0){max_moment2 = moment2;}
            else{max_moment2 = max_moment2.max(moment2);}
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1.array()*((max_moment2+std::pow(10,-10)).rsqrt()));
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();

        }
        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;

            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }

            moment1 = (1-beta1)*moment1 + beta1*gradient;
            moment2 = (1-beta2)*moment2 + beta2*gradient.array().pow(2);
            if(iter==0){max_moment2 = moment2;}
            else{max_moment2 = max_moment2.max(moment2);}
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1.array()*((max_moment2+std::pow(10,-10)).rsqrt()));
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();

        }


        moyGradientNorm = sommeGradient/number_batch;
        if(numericalNoise(moyGradientNorm)){break;}

        iter++;
    }

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    Sdouble cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;

    return study;
}

std::map<std::string,Sdouble> train_SGD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& clip, int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;

    if(algo=="SGD")
    {
        study = SGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="SGD_Ito")
    {
        study = SGD_Ito(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,eps,maxIter,record,fileExtension);
    }
    else if(algo=="SGD_clipping")
    {
        study = SGD_clipping(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,clip,batch_size,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="Momentum_Euler")
    {
        study = Momentum_Euler(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta1,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="Momentum")
    {
        study = Momentum(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta1,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="AdaGrad")
    {
        study = AdaGrad(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,eps,maxIter,record,fileExtension);
    }
    else if(algo=="RMSProp")
    {
        study = RMSProp(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta2,eps,maxIter,record,fileExtension);
    }
    else if(algo=="Adam")
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta1,beta2,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="AMSGrad")
    {
        study = AMSGrad(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta1,beta2,eps,maxIter,record,fileExtension);
    }
    else
    {
        study = SGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,eps,maxIter,record,tracking,fileExtension);
    }

    return study;

}

