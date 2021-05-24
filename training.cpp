#include "training.h"

std::map<std::string,double> LM(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<std::string> const& activations,
std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double mu, double const factor, double const eps, int const maxIter, bool const record, std::string fileExtension)
{


    std::ofstream weightsFlux(("Record/weights_LM_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LM_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LM_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    assert (factor>1);

    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N,N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    double cost = 0.5*E.squaredNorm(), costPrec;
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
    H = Q+mu*I;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    while (gradient.norm()>eps && iter<maxIter)
    {
        costPrec=cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost=0.5*E.squaredNorm();

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if (cost<costPrec)
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            mu/=factor;
            notBack++;
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            mu*=factor;
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        H = Q+mu*I;
        solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;

}

std::map<std::string,double> LMBall(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double mu, double factor, double const eps, int const maxIter, std::string const norm, double const radiusBall, bool const record, std::string const fileExtension)
{


    std::ofstream weightsFlux(("Record/weights_LMBall_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMBall_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMBall_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    assert (factor>1);

    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N,N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    double cost = 0.5*E.squaredNorm(), costPrec, dis;
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
    H = Q+mu*I;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    while (gradient.norm()>eps && iter<maxIter)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();

        dis = distance(weightsPrec,biasPrec,weights,bias,norm);
        if(dis>radiusBall)
        {
            if(radiusBall>1){factor=dis;}
            else if(dis<1){factor=1.0/radiusBall;}
            else if(dis>1 && radiusBall<1){factor=std::max(dis,1.0/radiusBall);}
        }

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

       if(cost>costPrec || dis>radiusBall)
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            mu*=factor;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }
        else
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            mu*=factor;
            notBack++;
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
        }

        H = Q+mu*I;
        solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;

}

std::map<std::string,double> LMF(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const eps, int const maxIter,
double const RMin, double const RMax, bool const record, std::string fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_LMF_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMF_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMF_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd D(N,N);
    double factor, linearReduction, R, muc, intermed;
    double mu=0;
    //mu=1; muc=0.75;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    double cost = 0.5*E.squaredNorm(), costPrec;
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
    scalingFletcher(Q,D,N);
    H = Q+mu*D;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    while (gradient.norm()>eps && iter<maxIter)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();
        intermed = delta.transpose()*gradient;
        linearReduction = -2*intermed-delta.transpose()*Q*delta;
        R = 2*(costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
             costFlux << cost << std::endl;
             muFlux << mu << std::endl;
        }

        if(R>RMax)
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(R<RMin)
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(factor<2){factor = 2;}
            if(factor>10){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }

        if (cost<costPrec)
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        H = Q+mu*D;
        solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;

}


std::map<std::string,double> LMMore(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps, int const maxIter, double const sigma, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LMMore_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMMore_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMMore_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N), q(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd D(N,N);
    double gamma, factor, linearReduction, R, Delta=0.5, upperBound, lowerBound, phi, phiDerivative, intermed;
    int iterSearch=0, maxIterSearch=10;
    double mu=0;
    //mu=1; muc=0.75;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    double cost = 0.5*E.squaredNorm(), costPrec;
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
    D=gradient.cwiseAbs().asDiagonal();
    upperBound = ((D.inverse()*gradient).norm())/Delta;
    if(std::abs(Q.determinant())<std::pow(10,-16)){lowerBound=0;}
    else
    {
        solve(gradient,Q,delta); q = D*delta;
        phi = q.norm()-Delta; solve(D*q,Q,delta); phiDerivative = ((D*q).transpose()*delta); phiDerivative/=q.norm(); lowerBound = -phi/phiDerivative;
    }
    if((D*gradient).norm()<(1+sigma)*Delta)
    {
        mu=0;
        delta=-gradient;
    }
    else
    {
        if (mu<lowerBound || mu>upperBound){mu=std::min(0.001*upperBound,std::sqrt(lowerBound*upperBound));}
        H = Q+mu*(D*D); solve(gradient,H,delta); q = D*delta;
        phi = q.norm()-Delta; solve(D*q,H,delta); phiDerivative = (D*q).transpose()*delta; phiDerivative/=q.norm();
        while (std::abs(phi)>sigma*Delta)
        {
            mu-=((phi+Delta)/Delta)*(phi/phiDerivative);
            H = Q+mu*(D*D); solve(gradient,H,delta); q = D*delta;
            phi = q.norm()-Delta; solve(D*q,H,delta); phiDerivative = ((D*q).transpose()*delta); phiDerivative/=q.norm();
            iterSearch++;
        }
        iterSearch=0;
        solve(gradient,H,delta);
    }

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    while (gradient.norm()>eps && iter<maxIter)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();
        intermed = delta.transpose()*gradient; linearReduction = -2*intermed-delta.transpose()*Q*delta;
        R = 2*(costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if (R<0.0001)
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }
        else
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
        }

        if (R<0.25)
        {
            gamma = intermed/(2*costPrec);
            factor = (0.5*gamma)/(gamma+0.5*(1-cost/costPrec));
            if(factor<0.1){factor=0.1;}
            else if(factor>0.5){factor=0.5;}
            else if(cost<costPrec){factor=0.5;}
            Delta*=factor;
        }
        else if ((R>0.25 && R<0.75 && mu<std::pow(10,-16))|| R>0.75)
        {
            Delta = 2*(D*delta).norm();
        }
        scalingMore(gradient,D);

        if ((D*gradient).norm()<(1+sigma)*Delta)
        {
            mu=0;
            delta=-gradient;
        }
        else
        {
            if (mu<lowerBound || mu>upperBound){mu=std::max(0.001*upperBound,std::sqrt(lowerBound*upperBound));}
            H = Q+mu*(D*D); solve(gradient,H,delta); q = D*delta;
            phi = q.norm()-Delta; solve(D*q,H,delta); phiDerivative = (D*q).transpose()*delta; phiDerivative/=q.norm();
            if (phi<0){upperBound=mu;}
            lowerBound = std::max(lowerBound,mu-phi/phiDerivative);

            while (std::abs(phi)>sigma*Delta && iterSearch<maxIterSearch)
            {
                mu-=((phi+Delta)/Delta)*(phi/phiDerivative);
                H = Q+mu*(D*D); solve(gradient,H,delta); q = D*delta;
                phi = q.norm()-Delta; solve(D*q,H,delta); phiDerivative = ((D*q).transpose()*delta); phiDerivative/=q.norm();
                iterSearch++;
            }
            iterSearch=0;
            solve(gradient,H,delta);
        }

        update(L,nbNeurons,globalIndices,weights,bias,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;
}

std::map<std::string,double> LMNielson(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps, int const maxIter, double const tau, double const beta, double const gamma, int const p, double const epsDiag,
bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LMNielson_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMNielson_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMNielson_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd I  = Eigen::MatrixXd::Identity(N,N);
    double mu, linearReduction, R, nu=beta, intermed;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    double cost = 0.5*E.squaredNorm(), costPrec;
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
    mu=tau*Q.diagonal().maxCoeff();
    H = Q+mu*I;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    while (gradient.norm()>eps && iter<maxIter)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();
        intermed = delta.transpose()*gradient;
        linearReduction = -2*intermed-delta.transpose()*Q*delta;
        R = 2*(costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if (R>0)
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            mu*=std::max(1.0/gamma,1-(beta-1)*std::pow(2*R-1,p));
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
            mu*=nu; nu*=2;
        }

        H = Q+mu*I;
        solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;
}

std::map<std::string,double> LMUphill(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const eps, int const maxIter,
double const RMin, double const RMax, int const b, bool const record, std::string fileExtension)
{

    assert(b==1 || b==2);

    std::ofstream weightsFlux(("Record/weights_LMUphill_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMUphill_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMUphill_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N), deltaPrec(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd D(N,N);
    double factor, linearReduction, R, muc, intermed;
    double mu=0;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    double cost = 0.5*E.squaredNorm(), costPrec; double costMin=cost;
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
    scalingFletcher(Q,D,N);
    H = Q+mu*D;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    while (gradient.norm()>eps && iter<maxIter && delta.lpNorm<Eigen::Infinity>()>eps*0.0001)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();
        intermed = delta.transpose()*gradient;
        linearReduction = -2*intermed-delta.transpose()*Q*delta;
        R = 2*(costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if(R>RMax)
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(R<RMin)
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(factor<2){factor = 2;}
            if(factor>10){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }
        if (cost<costPrec || (iter>1 && std::pow(1-cosVector(deltaPrec,delta),b)*cost<costMin))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            costMin=std::min(costMin,cost);
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        H = Q+mu*D;
        deltaPrec=delta; solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;

}

std::map<std::string,double> LMPerso(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const eps, int const maxIter,
double const RMin, double const RMax, int const b, double const epsDiag, bool const record, std::string fileExtension)
{

    assert(b==1 || b==2);

    std::ofstream weightsFlux(("Record/weights_LMPerso_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMPerso_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMPerso_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nbNeurons[L],P);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N), deltaPrec(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd D(N,N);
    double factor, linearReduction, R, muc, intermed;
    double mu=0;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    double cost = 0.5*E.squaredNorm(), costPrec; double costMin=cost;
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
    D=Q.diagonal().cwiseAbs().cwiseMax(epsDiag).asDiagonal();
    H = Q+mu*D;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);

    while (gradient.norm()>eps && iter<maxIter && delta.lpNorm<Eigen::Infinity>()>eps*0.0001)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();
        intermed = delta.transpose()*gradient;
        linearReduction = -2*intermed-delta.transpose()*Q*delta;
        R = 2*(costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if(R>RMax)
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(R<RMin)
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(factor<2){factor = 2;}
            if(factor>10){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }
        if (cost<costPrec || (iter>1 && std::pow(1-cosVector(deltaPrec,delta),b)*cost<costMin))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            costMin=std::min(costMin,cost);
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
            scalingMore2(Q,D,epsDiag);
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        H = Q+mu*D;
        deltaPrec=delta; solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;

}


std::map<std::string,double> LMGeodesic(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const eps, int const maxIter,
double const RMin, double const RMax, int const b, double const alpha, double const pas, bool const record, std::string fileExtension)
{

    assert(b==1 || b==2);

    std::ofstream weightsFlux(("Record/weights_LMGeodesic_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMGeodesic_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMGeodesic_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), nL=nbNeurons[L], iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nL,P), ETranspose(P,nL), EInter(nL,P), EInterTranspose(P,nL);
    Eigen::VectorXd gradient(N), Epp(P*nL);
    Eigen::MatrixXd Q (N,N);
    Eigen::MatrixXd J(P*nL,N);
    Eigen::VectorXd delta(N), delta1(N), delta1Prec(N), delta2(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd D(N,N);
    double factor, linearReduction, R, muc, intermed;
    double mu=0;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1);
    double cost = 0.5*E.squaredNorm(), costPrec; double costMin=cost;
    backwardJacob(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,J); Q=J.transpose()*J; gradient=J.transpose()*ETranspose;
    scalingFletcher(Q,D,N);
    H = Q+mu*D;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta1);
    update(L,nbNeurons,globalIndices,weights,bias,delta1);

    while (gradient.norm()>eps && iter<maxIter && delta1.lpNorm<Eigen::Infinity>()>eps*0.0001)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1);
        update(L,nbNeurons,globalIndices,weights,bias,pas*delta1); fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,EInter); EInterTranspose=EInter.transpose(); EInterTranspose.resize(nL*P,1);
        Epp=2.0/pas*(1.0/pas*(EInterTranspose-ETranspose)-J*delta1); solve(0.5*J.transpose()*Epp,H,delta2);
        delta=delta1+delta2;
        cost = 0.5*E.squaredNorm();
        intermed = delta1.transpose()*gradient;
        linearReduction = -2*intermed-delta1.transpose()*Q*delta1;
        R = 2*(costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if(R>RMax)
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(R<RMin)
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(factor<2){factor = 2;}
            if(factor>10){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }
        if ((cost<costPrec && 2*delta2.norm()/delta1.norm()<alpha) || (iter>1 && std::pow(1-cosVector(delta1Prec,delta1),b)*cost<costMin))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            notBack++;
            costMin=std::min(costMin,cost);
            backwardJacob(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,J); Q=J.transpose()*J; gradient=J.transpose()*ETranspose;
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        H = Q+mu*D;
        delta1Prec=delta1; solve(gradient,H,delta1);
        update(L,nbNeurons,globalIndices,weights,bias,delta1);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;

}

std::map<std::string,double> LMJynian(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
double const eps, int const maxIter, double const Rlim, double const RMin, double const RMax, double const factorMin, double const power, double const alphaChap,
bool const record, std::string const fileExtension)
{
    assert(0<power && power<4);

    std::ofstream weightsFlux(("Record/weights_LMJynian_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMJynian_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMJynian_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), nL=nbNeurons[L], iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::MatrixXd E(nL,P), ETranspose(P,nL);
    Eigen::VectorXd gradient(N), gradientChap(N);
    Eigen::MatrixXd Q(N,N);
    Eigen::MatrixXd J(P*nL,N);
    Eigen::VectorXd delta(N),deltaChap(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N,N);
    double mu=1.0, factor=factorMin, linearReduction, R, alpha, inter1,inter2,inter3,inter4;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1);
    double cost = 0.5*E.squaredNorm(), costPrec; double costMin=cost;
    backwardJacob(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,J); Q=J.transpose()*J; gradient=J.transpose()*ETranspose; H = Q+mu*I;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }

    solve(gradient,H,delta);
    update(L,nbNeurons,globalIndices,weights,bias,delta);
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1); gradientChap=J.transpose()*ETranspose;
    solve(gradientChap,H,deltaChap);
    inter4=deltaChap.transpose()*Q*deltaChap;
    if (std::abs(inter4)>std::pow(10,-16)){alpha=1+(mu*deltaChap.squaredNorm())/(inter4); alpha=std::min(alpha,alphaChap);}
    else {alpha=(2*alphaChap)*deltaChap.transpose()*H*deltaChap;}
    update(L,nbNeurons,globalIndices,weights,bias,alpha*deltaChap);

    while (gradient.norm()>eps && iter<maxIter && (delta+alpha*deltaChap).lpNorm<Eigen::Infinity>()>0.0001*eps)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1);
        cost = 0.5*E.squaredNorm();
        inter1=delta.transpose()*gradient; inter2=delta.transpose()*Q*delta; inter3=deltaChap.transpose()*gradientChap;
        linearReduction = -2*inter1-inter2-2*alpha*inter3-std::pow(alpha,2)*inter4;
        R = 2*(costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if(R<RMin){factor*=4;}
        else if(R>RMax){factor=std::max(factorMin,factor/4.0);}
        mu=factor*std::pow(2*cost,power/2.0);

        if (R>Rlim)
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            notBack++;
            costMin=std::min(costMin,cost);
            backwardJacob(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,J); Q=J.transpose()*J; gradient=J.transpose()*ETranspose;
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        H = Q+mu*I;
        solve(gradient,H,delta); update(L,nbNeurons,globalIndices,weights,bias,delta);
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1); gradientChap=J.transpose()*ETranspose;
        solve(gradientChap,H,deltaChap);
        inter4=deltaChap.transpose()*Q*deltaChap;
        if (std::abs(inter4)>std::pow(10,-16)){alpha=1+(mu*deltaChap.squaredNorm())/(inter4); alpha=std::min(alpha,alphaChap);}
        else {alpha=(2*alphaChap)*deltaChap.transpose()*H*deltaChap;}
        update(L,nbNeurons,globalIndices,weights,bias,alpha*deltaChap);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,double> study;
    study["iter"]=(double)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(iter-notBack); study["propBack"]=(double)nbBack/(double)iter;

    return study;
}

std::map<std::string,double> train(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const algo, double const eps, int const maxIter,
double mu, double const factor, double const RMin, double const RMax, int const b, double const alpha, double const pas, double const Rlim,
double const factorMin, double const power, double const alphaChap, double const epsDiag, double const tau, double const beta,
double const gamma, int const p, double const sigma, std::string const norm, double const radiusBall, bool const record, std::string const fileExtension)
{
    std::map<std::string,double> study;

    if(algo=="LM"){study = LM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter,record,fileExtension);}
    else if(algo=="LMF"){study = LMF(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,record,fileExtension);}
    else if(algo=="LMUphill"){study = LMUphill(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b,record,fileExtension);}
    else if(algo=="LMGeodesic"){study = LMGeodesic(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b,alpha,pas,record,fileExtension);}
    else if(algo=="LMJynian"){study = LMJynian(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,Rlim,RMin,RMax,factorMin,power,alphaChap,record,fileExtension);}
    else if(algo=="LMMore"){study = LMMore(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,sigma,record,fileExtension);}
    else if(algo=="LMNielson"){study = LMNielson(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,tau,beta,gamma,p,epsDiag,record,fileExtension);}
    else if(algo=="LMBall"){study = LMBall(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter,norm,radiusBall,record,fileExtension);}

    return study;
}

