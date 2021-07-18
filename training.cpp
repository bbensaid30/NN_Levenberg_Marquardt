#include "training.h"

std::map<std::string,Sdouble> LM(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, Sdouble& mu, Sdouble& factor, Sdouble const& eps,
int const& maxIter, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LM_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LM_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LM_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    assert (factor>1);

    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd I = Eigen::SMatrixXd::Identity(N,N);

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);


    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    Sdouble cost = 0.5*E.squaredNorm(), costPrec;
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

    Sdouble gradientNorm = gradient.norm();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
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

        if (std::signbit((cost-costPrec).number))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            mu/=factor;
            notBack++;
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            mu*=factor;
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu)){break;}

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

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=Sdouble(endSequenceMax-notBackMax);
    study["endSequenceMax"]=Sdouble(endSequenceMax); study["startSequenceFinal"]=Sdouble(iter-notBack); study["propBack"]=Sdouble(nbBack)/Sdouble(iter);

    return study;

}

std::map<std::string,Sdouble> LMBall(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Sdouble& mu, Sdouble& factor, Sdouble const& eps, int const& maxIter, std::string const& norm, Sdouble const& radiusBall, bool const record, std::string const fileExtension)
{


    std::ofstream weightsFlux(("Record/weights_LMBall_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMBall_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMBall_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    assert (factor>1);

    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd I = Eigen::SMatrixXd::Identity(N,N);

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    Sdouble cost = 0.5*E.squaredNorm(), costPrec, dis;
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

    Sdouble gradientNorm = gradient.norm();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();

        dis = distance(weightsPrec,biasPrec,weights,bias,norm);
        if(!std::signbit((dis-radiusBall).number))
        {
            if(!std::signbit((radiusBall-1).number)){factor=dis;}
            else if(std::signbit((dis-1).number)){factor=1.0/radiusBall;}
            else if(!std::signbit((dis-1).number) && std::signbit((radiusBall-1).number)){factor=Sstd::max(dis,1.0/radiusBall);}
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

       if(!std::signbit((cost-costPrec).number) || !std::signbit((dis-radiusBall).number))
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
            gradientNorm = gradient.norm();
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(factor)){break;}

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

    std::map<std::string,Sdouble> study;
    study["iter"]=(Sdouble)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(Sdouble)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(Sdouble)endSequenceMax; study["startSequenceFinal"]=(Sdouble)(iter-notBack); study["propBack"]=(Sdouble)nbBack/(Sdouble)iter;

    return study;

}

std::map<std::string,Sdouble> LMF(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, Sdouble const& eps, int const& maxIter,
Sdouble const& RMin, Sdouble const& RMax, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_LMF_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMF_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMF_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd D(N,N);
    Sdouble factor, linearReduction, R, mu=10, muc=0, intermed;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    Sdouble cost = 0.5*E.squaredNorm(), costPrec;
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


    Sdouble gradientNorm = gradient.norm();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();
        intermed = delta.transpose()*gradient;
        linearReduction = -delta.transpose()*Q*delta; linearReduction-=2*intermed;
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

        if(!std::signbit((R-RMax).number))
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(std::signbit((R-RMin).number))
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(std::signbit((factor-2).number)){factor = 2;}
            if(!std::signbit((factor-10).number)){factor = 10;}

            if(mu<std::pow(10,-16))
            {
//                Eigen::SelfAdjointEigenSolver<Eigen::SMatrixXd> es(Q);
//                muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }

        if (std::signbit((cost-costPrec).number))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(factor)){break;}

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

    std::map<std::string,Sdouble> study;
    study["iter"]=(Sdouble)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(Sdouble)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(Sdouble)endSequenceMax; study["startSequenceFinal"]=(Sdouble)(iter-notBack); study["propBack"]=(Sdouble)nbBack/(Sdouble)iter;

    return study;

}


std::map<std::string,Sdouble> LMMore(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Sdouble const& eps, int const& maxIter, Sdouble const& sigma, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LMMore_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMMore_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMMore_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N), q(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd D(N,N);
    Sdouble gamma, factor, linearReduction, R, Delta=0.5, upperBound, lowerBound, phi, phiDerivative, intermed;
    int iterSearch=0, maxIterSearch=10;
    Sdouble mu=0;
    Sdouble detQ, DgradientNorm;
    //mu=1; muc=0.75;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    Sdouble cost = 0.5*E.squaredNorm(), costPrec;
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
    D=gradient.cwiseAbs().asDiagonal();
    upperBound = ((D.inverse()*gradient).norm())/Delta;
    detQ = Sstd::abs(Q.determinant());
    if(detQ+std::abs(detQ.error)<std::pow(10,-16)){lowerBound=0;}
    else
    {
        solve(gradient,Q,delta); q = D*delta;
        phi = q.norm()-Delta; solve(D*q,Q,delta); phiDerivative = ((D*q).transpose()*delta); phiDerivative/=q.norm(); lowerBound = -phi/phiDerivative;
    }
    DgradientNorm = (D*gradient).norm();
    if(DgradientNorm+std::abs(DgradientNorm.error)<(1+sigma)*Delta+std::abs(((1+sigma)*Delta).error))
    {
        mu=0;
        delta=-gradient;
    }
    else
    {
        if (mu+std::abs(mu.error)<lowerBound-std::abs(lowerBound.error) || mu-std::abs(mu.error)>upperBound+std::abs(upperBound.error)){mu=Sstd::min(0.001*upperBound,Sstd::sqrt(lowerBound*upperBound));}
        H = Q+mu*(D*D); solve(gradient,H,delta); q = D*delta;
        phi = q.norm()-Delta; solve(D*q,H,delta); phiDerivative = (D*q).transpose()*delta; phiDerivative/=q.norm();
        while (Sstd::abs(phi)-std::abs(phi.error)>sigma*Delta+std::abs((sigma*Delta).error))
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

    Sdouble gradientNorm = gradient.norm();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();
        intermed = delta.transpose()*gradient; linearReduction = -delta.transpose()*Q*delta; linearReduction-=2*intermed;
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

        if (R+std::abs(R.error)<0.0001)
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
            gradientNorm = gradient.norm();
        }

        if (R+std::abs(R.error)<0.25)
        {
            gamma = intermed/(2*costPrec);
            factor = (0.5*gamma)/(gamma+0.5*(1-cost/costPrec));
            if(factor+std::abs(factor.error)<0.1){factor=0.1;}
            else if(factor-std::abs(factor.error)>0.5){factor=0.5;}
            else if(cost+std::abs(cost.error) < costPrec-std::abs(costPrec.error)){factor=0.5;}
            Delta*=factor;
        }
        else if ((R-std::abs(R.error)>0.25 && R<0.75 && mu+std::abs(mu.error)<std::pow(10,-16))|| R-std::abs(R.error)>0.75)
        {
            Delta = 2*(D*delta).norm();
        }
        scalingMore(gradient,D);

        DgradientNorm = (D*gradient).norm();
        if (DgradientNorm+std::abs(DgradientNorm.error)<(1+sigma)*Delta+std::abs(((1+sigma)*Delta).error))
        {
            mu=0;
            delta=-gradient;
        }
        else
        {
            if (mu+std::abs(mu.error)<lowerBound-std::abs(lowerBound.error) || mu-std::abs(mu.error)>upperBound+std::abs(upperBound.error)){mu=Sstd::max(0.001*upperBound,Sstd::sqrt(lowerBound*upperBound));}
            H = Q+mu*(D*D); solve(gradient,H,delta); q = D*delta;
            phi = q.norm()-Delta; solve(D*q,H,delta); phiDerivative = (D*q).transpose()*delta; phiDerivative/=q.norm();
            if (phi+std::abs(phi.error)<0){upperBound=mu;}
            lowerBound = Sstd::max(lowerBound,mu-phi/phiDerivative);

            while (Sstd::abs(phi)-std::abs(phi.error)>sigma*Delta+std::abs((sigma*Delta).error) && iterSearch<maxIterSearch)
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

    std::map<std::string,Sdouble> study;
    study["iter"]=(Sdouble)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(Sdouble)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(Sdouble)endSequenceMax; study["startSequenceFinal"]=(Sdouble)(iter-notBack); study["propBack"]=(Sdouble)nbBack/(Sdouble)iter;

    return study;
}

std::map<std::string,Sdouble> LMNielson(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Sdouble const& eps, int const& maxIter, Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& epsDiag,
bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LMNielson_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMNielson_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMNielson_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd I  = Eigen::SMatrixXd::Identity(N,N);
    Sdouble mu, linearReduction, R, nu=beta, intermed;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    Sdouble cost = 0.5*E.squaredNorm(), costPrec;
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

    Sdouble gradientNorm = gradient.norm();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();
        intermed = delta.transpose()*gradient;
        linearReduction = -delta.transpose()*Q*delta; linearReduction-=2*intermed;
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

        if (!std::signbit(R.number))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            mu*=maximum(1.0/gamma,1-(beta-1)*Sstd::pow(2*R-1,p));
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
            mu*=nu; nu*=2;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(nu)){break;}

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

    std::map<std::string,Sdouble> study;
    study["iter"]=(Sdouble)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(Sdouble)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(Sdouble)endSequenceMax; study["startSequenceFinal"]=(Sdouble)(iter-notBack); study["propBack"]=(Sdouble)nbBack/(Sdouble)iter;

    return study;
}

std::map<std::string,Sdouble> LMUphill(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, Sdouble const& eps, int const& maxIter,
Sdouble const& RMin, Sdouble const& RMax, int const& b, bool const record, std::string const fileExtension)
{

    assert(b==1 || b==2);

    std::ofstream weightsFlux(("Record/weights_LMUphill_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMUphill_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMUphill_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N), deltaPrec(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd D(N,N);
    Sdouble factor, linearReduction, R, mu=10, muc=0, intermed;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    Sdouble cost = 0.5*E.squaredNorm(), costPrec; Sdouble costMin=cost;
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

    Sdouble gradientNorm = gradient.norm();
    Sdouble deltaNorm = delta.norm();
    Sdouble angle;

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter && deltaNorm+std::abs(deltaNorm.error)>eps*std::pow(10,-3))
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();
        intermed = delta.transpose()*gradient;
        linearReduction = -delta.transpose()*Q*delta; linearReduction-=2*intermed;
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

        if(!std::signbit((R-RMax).number))
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(std::signbit((R-RMin).number))
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(std::signbit((factor-2).number)){factor = 2;}
            if(!std::signbit((factor-10).number)){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::SMatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }
        angle = Sstd::pow(1-cosVector(deltaPrec,delta),b)*cost;
        if (std::signbit((cost-costPrec).number) || (iter>1 && std::signbit((angle-costMin).number)))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            costMin=minimum(costMin,cost);
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(factor)){break;}

        H = Q+mu*D;
        deltaPrec=delta; solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        deltaNorm = delta.norm();
        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,Sdouble> study;
    study["iter"]=(Sdouble)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(Sdouble)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(Sdouble)endSequenceMax; study["startSequenceFinal"]=(Sdouble)(iter-notBack); study["propBack"]=(Sdouble)nbBack/(Sdouble)iter;

    return study;

}

std::map<std::string,Sdouble> LMPerso(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, Sdouble const& eps, int const& maxIter,
Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& epsDiag, bool const record, std::string const fileExtension)
{

    assert(b==1 || b==2);

    std::ofstream weightsFlux(("Record/weights_LMPerso_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMPerso_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMPerso_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N), deltaPrec(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd D(N,N);
    Sdouble factor, linearReduction, R, mu=10, muc=0, intermed;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    Sdouble cost = 0.5*E.squaredNorm(), costPrec; Sdouble costMin=cost;
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

    Sdouble gradientNorm = gradient.norm();
    Sdouble deltaNorm = delta.lpNorm<Eigen::Infinity>();
    Sdouble angle;

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter && deltaNorm+std::abs(deltaNorm.error)>eps*std::pow(10,-3))
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
        cost = 0.5*E.squaredNorm();
        intermed = delta.transpose()*gradient;
        linearReduction = -delta.transpose()*Q*delta; linearReduction-=2*intermed;
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

        if(!std::signbit((R-RMax).number))
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(std::signbit((R-RMin).number))
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(std::signbit((factor-2).number)){factor = 2;}
            if(!std::signbit((factor-10).number)){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::SMatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }

        angle = Sstd::pow(1-cosVector(deltaPrec,delta),b)*cost;
        if (std::signbit((cost-costPrec).number) || (iter>1 && std::signbit((angle-costMin).number)))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            costMin=minimum(costMin,cost);
            backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
            scalingMore2(Q,D,epsDiag);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(factor)){break;}

        H = Q+mu*D;
        deltaPrec=delta; solve(gradient,H,delta);
        update(L,nbNeurons,globalIndices,weights,bias,delta);

        deltaNorm = delta.lpNorm<Eigen::Infinity>();
        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,Sdouble> study;
    study["iter"]=(Sdouble)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(Sdouble)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(Sdouble)endSequenceMax; study["startSequenceFinal"]=(Sdouble)(iter-notBack); study["propBack"]=(Sdouble)nbBack/(Sdouble)iter;

    return study;

}


std::map<std::string,Sdouble> LMGeodesic(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, Sdouble const& eps, int const& maxIter,
Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha, Sdouble const& pas, bool const record, std::string const fileExtension)
{

    assert(b==1 || b==2);

    std::ofstream weightsFlux(("Record/weights_LMGeodesic_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMGeodesic_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMGeodesic_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), nL=nbNeurons[L], iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nL,P), ETranspose(P,nL), EInter(nL,P), EInterTranspose(P,nL);
    Eigen::SVectorXd gradient(N), Epp(P*nL);
    Eigen::SMatrixXd Q (N,N);
    Eigen::SMatrixXd J(P*nL,N);
    Eigen::SVectorXd delta(N), delta1(N), delta1Prec(N), delta2(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd D(N,N);
    Sdouble factor, linearReduction, R, mu=10, muc=0, intermed;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1);
    Sdouble cost = 0.5*E.squaredNorm(), costPrec; Sdouble costMin=cost;
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

    Sdouble gradientNorm = gradient.norm();
    Sdouble delta1Norm = delta1.norm();
    Sdouble angle, quotient;

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter && delta1Norm+std::abs(delta1Norm.error)>eps*std::pow(10,-3))
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1);
        update(L,nbNeurons,globalIndices,weights,bias,pas*delta1); fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,EInter); EInterTranspose=EInter.transpose(); EInterTranspose.resize(nL*P,1);
        Epp=2.0/pas*(1.0/pas*(EInterTranspose-ETranspose)-J*delta1); solve(0.5*J.transpose()*Epp,H,delta2);
        delta=delta1+delta2;
        cost = 0.5*E.squaredNorm();
        intermed = delta1.transpose()*gradient;
        linearReduction = -delta1.transpose()*Q*delta1; linearReduction-=2*intermed;
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

        if(!std::signbit((R-RMax).number))
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(std::signbit((R-RMin).number))
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(std::signbit((factor-2).number)){factor = 2;}
            if(!std::signbit((factor-10).number)){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::SMatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }

        angle = Sstd::pow(1-cosVector(delta1Prec,delta1),b)*cost;
        quotient = 2*delta2.norm()/delta1.norm();

        if ((std::signbit((cost-costPrec).number) && std::signbit((quotient-alpha).number)) || (iter>1 && std::signbit((angle-costMin).number)))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            notBack++;
            costMin=minimum(costMin,cost);
            backwardJacob(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,J); Q=J.transpose()*J; gradient=J.transpose()*ETranspose;
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(factor)){break;}

        H = Q+mu*D;
        delta1Prec=delta1; solve(gradient,H,delta1);
        update(L,nbNeurons,globalIndices,weights,bias,delta1);

        delta1Norm = delta1.norm();

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,Sdouble> study;
    study["iter"]=(Sdouble)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(Sdouble)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(Sdouble)endSequenceMax; study["startSequenceFinal"]=(Sdouble)(iter-notBack); study["propBack"]=(Sdouble)nbBack/(Sdouble)iter;

    return study;

}

std::map<std::string,Sdouble> LMJynian(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias,
Sdouble const& eps, int const& maxIter, Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap,
bool const record, std::string const fileExtension)
{
    assert(0<power && power<4);

    std::ofstream weightsFlux(("Record/weights_LMJynian_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMJynian_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMJynian_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), nL=nbNeurons[L], iter=1, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nL,P), ETranspose(P,nL);
    Eigen::SVectorXd gradient(N), gradientChap(N);
    Eigen::SMatrixXd Q(N,N);
    Eigen::SMatrixXd J(P*nL,N);
    Eigen::SVectorXd delta(N),deltaChap(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd I = Eigen::SMatrixXd::Identity(N,N);
    Sdouble mu=1.0, factor=factorMin, linearReduction, R, alpha, inter1,inter2,inter3,inter4;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1);
    Sdouble cost = 0.5*E.squaredNorm(), costPrec; Sdouble costMin=cost;
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
    if (Sstd::abs(inter4)>std::pow(10,-16)){alpha=1+(mu*deltaChap.squaredNorm())/(inter4); alpha=Sstd::min(alpha,alphaChap);}
    else {alpha=(2*alphaChap)*deltaChap.transpose()*H*deltaChap;}
    update(L,nbNeurons,globalIndices,weights,bias,alpha*deltaChap);

    Sdouble gradientNorm = gradient.norm();
    Sdouble deltaCNorm = (delta+alpha*deltaChap).lpNorm<Eigen::Infinity>();

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter && deltaCNorm+std::abs(deltaCNorm.error)>eps*std::pow(10,-3))
    {
        costPrec = cost;
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1);
        cost = 0.5*E.squaredNorm();
        inter1=delta.transpose()*gradient; inter2=delta.transpose()*Q*delta; inter3=deltaChap.transpose()*gradientChap;
        linearReduction = -2*inter1-inter2-2*alpha*inter3-Sstd::pow(alpha,2)*inter4;
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

        if(std::signbit((R-RMin).error)){factor*=4;}
        else if(!std::signbit((R-RMax).number)){factor=Sstd::max(factorMin,factor/4.0);}
        mu=factor*Sstd::pow(2*cost,power/2.0);

        if (!std::signbit((R-Rlim).number))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            notBack++;
            costMin=Sstd::min(costMin,cost);
            backwardJacob(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,J); Q=J.transpose()*J; gradient=J.transpose()*ETranspose;
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(factor)){break;}

        H = Q+mu*I;
        solve(gradient,H,delta,"LeastSquaresConjugateGradient"); update(L,nbNeurons,globalIndices,weights,bias,delta);
        fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E); ETranspose=E.transpose(); ETranspose.resize(P*nL,1); gradientChap=J.transpose()*ETranspose;
        solve(gradientChap,H,deltaChap,"LeastSquaresConjugateGradient");
        inter4=deltaChap.transpose()*Q*deltaChap; if(numericalNoise(inter4)){break;}
        if (Sstd::abs(inter4)>std::pow(10,-16)){alpha=1+(mu*deltaChap.squaredNorm())/(inter4); alpha=Sstd::min(alpha,alphaChap);}
        else {alpha=(2*alphaChap)*deltaChap.transpose()*H*deltaChap;}
        update(L,nbNeurons,globalIndices,weights,bias,alpha*deltaChap);

        deltaCNorm = (delta+alpha*deltaChap).lpNorm<Eigen::Infinity>();
        iter++;
    }

    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    cost = 0.5*E.squaredNorm();

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=Sdouble(endSequenceMax-notBackMax);
    study["endSequenceMax"]=Sdouble(endSequenceMax); study["startSequenceFinal"]=Sdouble(iter-notBack); study["propBack"]=Sdouble(nbBack)/Sdouble(iter);

    return study;
}

std::map<std::string,Sdouble> init(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons,std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LM_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LM_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(),l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SMatrixXd E(nbNeurons[L],P);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);

    fforward(X,Y,L,P,nbNeurons,activations,weights,bias,As,slopes,E);
    Sdouble cost = 0.5*E.squaredNorm();
    backward(L,P,nbNeurons,globalIndices,weights,bias,As,slopes,E,gradient,Q);
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

    std::map<std::string,Sdouble> study;
    study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;
    std::cout << "gradient: " << gradient.norm() << std::endl;

    return study;

}

std::map<std::string,Sdouble> train(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& algo, Sdouble const& eps, int const& maxIter,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha, Sdouble const& pas, Sdouble const& Rlim,
Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag, Sdouble const& tau, Sdouble const& beta,
Sdouble const& gamma, int const& p, Sdouble const& sigma, std::string const& norm, Sdouble const& radiusBall, bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;

    if(algo=="LM"){study = LM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter,record,fileExtension);}
    else if(algo=="LMF"){study = LMF(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,record,fileExtension);}
    else if(algo=="LMUphill"){study = LMUphill(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b,record,fileExtension);}
    else if(algo=="LMPerso"){study = LMPerso(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b,epsDiag,record,fileExtension);}
    else if(algo=="LMGeodesic"){study = LMGeodesic(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,RMin,RMax,b,alpha,pas,record,fileExtension);}
    else if(algo=="LMJynian"){study = LMJynian(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,Rlim,RMin,RMax,factorMin,power,alphaChap,record,fileExtension);}
    else if(algo=="LMMore"){study = LMMore(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,sigma,record,fileExtension);}
    else if(algo=="LMNielson"){study = LMNielson(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,eps,maxIter,tau,beta,gamma,p,epsDiag,record,fileExtension);}
    else if(algo=="LMBall"){study = LMBall(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,mu,factor,eps,maxIter,norm,radiusBall,record,fileExtension);}
    else if(algo=="init"){study = init(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,record,fileExtension);}

    return study;
}
