library(stats)
library(graphics)
library(cluster)
library(fpc)
library(dbscan)
library(data.table)

#------------------------------------ Load the data ---------------------------------------------------------------------------

algo <- 'LMF'
epsString <- '1e-10'
PString <- '40'
activationString <- 'sigmoidl'
folder <- paste0('sineWave/','P=40|width=1|', activationString)
fileExtension <- '1-'

fileNameParameter <- paste0('Record/',folder,'/weights_vectors_',algo,'_',fileExtension,'(eps=',epsString,', P=',PString,").csv")
fileNameCost <- paste0('Record/',folder,'/cost_',algo,'_',fileExtension,'(eps=',epsString,', P=',PString,").csv")
fileNameGradientNorm <- paste0('Record/',folder,'/gradientNorm_',algo,'_',fileExtension,'(eps=',epsString,', P=',PString,").csv")

parameters <- read.table(fileNameParameter,header=FALSE)
fileCosts <- read.table(fileNameCost,header=FALSE)
costs <- as.data.frame(fileCosts[seq(1, nrow(fileCosts), 3),])
colnames(costs) <- c("cost")
fileGradientNorms <- read.table(fileNameGradientNorm,header=FALSE)
gradientNorms <- as.data.frame(fileGradientNorms[seq(1, nrow(fileGradientNorms), 3),])
colnames(gradientNorms) <- c("gradientNorm")
points <- cbind(parameters,costs)

#------------------------------ Scaling and Normalization -------------------------------------------------------------------

normer <- function(points,typeNorm)
{
  nbData <- dim(points)[1]
  nbVar <- dim(points)[2]
  
  pointsNorm <- copy(points)
  
  for(i in 1:nbData)
  {
    norm <- norm(as.matrix(points[i,]),type=typeNorm)
    if(norm>10^(-20))
    {
      pointsNorm[i,] <- points[i,]/norm
    }
  }
  
  return (pointsNorm)
  
}

pointsScale <- scale(points)
pointsNorm <- normer(points,"2")
pointsScaleNorm <- scale(pointsNorm)

data <- pointsScaleNorm
nbData <- dim(data)[1]
nbVar <- dim(data)[2]

#----------------------------------------------- Kmeans ---------------------------------------------------------------------------------

nbClusterKmeans <- function(data,ecart,trace=FALSE)
{
  K <- c()
  J <- c()
  JJ <- c()
  inertie_prec <- 0
  k <- 2
  continuer <- TRUE
  nbData <- nrow(data)
  while (continuer & k < nbData)
  {
    cl <- kmeans(data,k)
    K <- c(K,k)
    J <- c(J, 1/nbData * cl$tot.withinss)
    xx <- data- cl$center[cl$cluster]
    JJ <- c(JJ, 1/nbData * sum(xx * xx))
    if(abs(inertie_prec-JJ[k-1]) <= ecart)
    {
      continuer <- FALSE
    }
    else
    {
      inertie_prec <- JJ[k-1]
      k <- k+1
    }
    print(k)
  }
  if(trace)
  {
    plot(K,JJ, type='l')
    points(K,J)
  }
  
  return (k)
}

nbClusterKmeansKMax <- function(data,KMax)
{
  K <- 2:KMax
  J <- matrix(0,length(KMax),1)
  JJ <- matrix(0,length(KMax),1)
  nbData <- nrow(data)
  for (k in K)
  {
    cl <- kmeans(data,k)
    J[k-1] <- 1/nbData * cl$tot.withinss
    xx <- data- cl$center[cl$cluster]
    JJ[k-1] <- 1/nbData * sum(xx * xx)
  }
 
  plot(K,JJ, type='l')
  points(K,J)
  
}

agregCluster <- function(data,ecart,nbTries)
{
  clusters <- matrix(0,nbTries,1)
  for(i in 1:nbTries)
  {
    clusters[i] <- nbClusterKmeans(data,ecart)
    print(clusters[i])
  }
  
  return (median(clusters))
}

ecart <- 10^(-7)
nbTries <- 100
nbClusterKmeansKMax(data,200)
nbCluster <- floor(agregCluster(data,ecart,nbTries))
nbCluster

cl <- kmeans(data,40)
plot(data, col=cl$cluster)
cl$centers
1/nbData*cl$tot.withinss

#------------------------------------------------------ Kmedoid ------------------------------------------------------------------------------

cp <- pam(data,nbCluster)
cp$medoids

#------------------------------------------------------ DBSCAN ---------------------------------------------------------------------------

minPts <- 10
dbscan::kNNdistplot(data, k =  minPts)
eps_db <- 10^(0)
db_x <- dbscan::dbscan(data,eps_db,minPts)
db_x
db_x$cluster
plot(data, col = db_x$cluster + 1)
title(paste0("DBSCAN pour eps=",eps_db,")"))

#-------------------------------------------------- Méthodes hiérarchiques ----------------------------------------------------------------

md <- dist(data)
hclust_methods <- c("ward.D", "single", "complete", "average", "mcquitty", 
                    "median", "centroid")
hh <- hclust(md, method = hclust_methods[3])   
plot(hh)

