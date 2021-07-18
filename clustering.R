library(stats)
library(graphics)
library(cluster)
library(fpc)
library(data.table)

#------------------------------------ Load the data ---------------------------------------------------------------------------

algo <- 'LMGeodesic'
epsString <- '1e-07'
fileExtension <- 'sinc1'

fileNameParameter <- paste0('Record/weights_vectors_',algo,'_',fileExtension,'(eps=',epsString,").csv")
fileNameCost <- paste0('Record/cost_',algo,'_',fileExtension,'(eps=',epsString,").csv")
fileNameGradientNorm <- paste0('Record/gradientNorm_',algo,'_',fileExtension,'(eps=',epsString,").csv")

parameters <- read.table(fileNameParameter,header=FALSE)
costs <- read.table(fileNameCost,header=FALSE)
colnames(costs) <- c("cost")
gradientNorms <- read.table(fileNameGradientNorm,header=FALSE)
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

data <- pointsNorm
nbData <- dim(data)[1]
nbVar <- dim(data)[2]

#----------------------------------------------- Kmeans ---------------------------------------------------------------------------------

nbClusterKmeans <- function(data,Kmax)
{
    K = 2:Kmax;
    J<- matrix(0,length(K),1);
    JJ<- matrix(0,length(K),1);
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
nbClusterKmeans(data,40)
nbCluster <- 4

cl <- kmeans(data,nbCluster)
plot(data, col=cl$cluster)
cl$centers
1/nbData*cl$tot.withinss

#------------------------------------------------------ Kmedoid ------------------------------------------------------------------------------

cp <- pam(data,nbCluster)
cp$medoids

#------------------------------------------------------ DBSCAN ---------------------------------------------------------------------------

eps_db <- 0.3
minPts <- 10
db_x <- dbscan(data,eps=eps_db,MinPts=minPts)
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

