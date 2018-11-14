# -*- coding: utf-8 -*-
"""
Created on Wed May 16 18:33:00 2018

@author: hexin
"""

import numpy as np
import random
import matplotlib.pyplot as plt 
import math

def loadDataSet(filename):
    fr = open(filename)
    dataMat = []
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataMat.append(list(map(float,line)))
    return dataMat

def distTwoVec(vec1,vec2):
    ##俩行矩阵求欧式距离
    return float(np.sqrt((vec1-vec2)*(vec1-vec2).T))

def randKpoint(dataMat,k):
    #dataMat为矩阵，k为聚类的个数，函数随机产生k个初始聚类中心点
    m = np.shape(dataMat)[0]
    index = random.sample(range(m),k)
    clusterCenter = dataMat[index]
    return clusterCenter

def randCent(dataSet, k):
    n =np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = np.float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))
    return centroids

#dataMat = loadDataSet('testSet.txt')
#dataMat = np.mat(dataMat)
#clu = randKpoint(dataMat,5)
    
def kMeans(dataMat,k,randCent = randCent,distTwoVec = distTwoVec):
    dataMat = np.mat(dataMat)
    clusterCen = randKpoint(dataMat,k)
    m,n = np.shape(dataMat)
    cluClass = np.mat(np.zeros((m,2)))
    cluContiue = True
    while cluContiue:
        cluContiue = False
        for i in range(m):
            shortDist = np.inf
            for j in range(k):
                distIJ = distTwoVec(dataMat[i,:],clusterCen[j,:])
                if distIJ < shortDist:
                    shortDist = distIJ
                    minIndex = j
            if cluClass[i,0] != minIndex:
                cluContiue = True  #当存在元素的分簇改变，继续迭代
            cluClass[i,:] = minIndex,shortDist**2
        for cent in range(k):#recalculate centroids
            ptsInClust = dataMat[np.nonzero(cluClass[:,0].A==cent)[0]]#得到每一个分类点
            clusterCen[cent,:] = np.mean(ptsInClust, axis=0)
    return clusterCen, cluClass            

def plotData(dataMat,clusterCen,cluClass,k):
    dataMat = np.mat(dataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    ax.scatter(dataMat[:,0].T.tolist()[0],dataMat[:,1].T.tolist()[0],\
#               s = ((cluClass[:,0]+1)*10).T.tolist()[0],c =((cluClass[:,0]+1)*30).T.tolist()[0])
    ax.scatter(dataMat[:,0].T.tolist()[0],dataMat[:,1].T.tolist()[0],\
              c =((cluClass[:,0]+1)*30).T.tolist()[0])
    ax.scatter(clusterCen[:,0].T.tolist()[0],clusterCen[:,1].T.tolist()[0],s =120,\
               c=list(np.arange(30,30*(k+1),30)),marker = 'x')

    plt.show()       
#dataMat = loadDataSet('testSet2.txt')
#clusterCen, cluClass = kMeans(dataMat,3)
#plotData(dataMat,clusterCen,cluClass,3)
#            
def biKmeans(dataSet, k, distMeas=distTwoVec):
    dataSet = np.mat(dataSet)
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0] #初始化为一个簇，中心为对应均值
    centList =[centroid0] #簇中心点列表
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distTwoVec=distMeas)
            sseSplit = np.sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
#            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
#        print ('the bestCentToSplit is: ',bestCentToSplit)
#        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return np.mat(centList), clusterAssment
   
#dataMat = loadDataSet('testSet2.txt')
#clusterCen, cluClass = biKmeans(dataMat,3)         
#plotData(dataMat,clusterCen,cluClass,3)

          
       

"""
高斯混合聚类，利用k均值聚类来初始化高斯聚类的值
"""
     
def mixGaussDisP(vecVal,vecMean,vecCor):
    ##均为np矩阵,定义多元变量的高斯分布,vecval是一个列向量
    n = np.shape(vecVal)[0]
    upp = np.exp(-0.5*(vecVal-vecMean).T*np.linalg.inv(vecCor)*(vecVal-vecMean))
    low = (2*np.pi)**(n/2)*np.linalg.det(vecCor)**(0.5)
    return float(upp/low)
    

def gammaVal(dataMat,gaussMean,gaussCor,alpha,k):
    m = np.shape(dataMat)[0]
    gamma = np.mat(np.zeros((m,k)))
    for j in range(m):
        low = sum([alpha[t]*mixGaussDisP(dataMat[j].T,\
               gaussMean[t],gaussCor[t]) for t in range(k)])
        for i in range(k):
            gamma[j,i] = alpha[i]*mixGaussDisP(dataMat[j].T,\
               gaussMean[i],gaussCor[i])/low
    return gamma


    

#    return gaussMean,gaussCor,alpha

def likehoodEst(dataMat,gaussMean,gaussCor,alpha,k):
    m,n = np.shape(dataMat)
    gamma = gammaVal(dataMat,gaussMean,gaussCor,alpha,k)
    for i in range(k):
        gaussMean[i] = (gamma[:,i].T*dataMat/(gamma[:,i].sum())).T
        gaussCor[i] = 0
        for j in range(m):
            gaussCor[i] += gamma[j,i]*(dataMat[j,:].T-gaussMean[i])*\
            (dataMat[j,:]-gaussMean[i].T)/(gamma[:,i].sum())
#        print(gamma[:,i].sum())
        alpha[i] = (gamma[:,i].sum())/m
    return gaussMean,gaussCor,alpha


def likeFunVal(dataMat,gaussMean,gaussCor,alpha,k):
    m,n = np.shape(dataMat)
    funVal = 1
    for j in range(m):
        likehoodVal = [float(alpha[i]*mixGaussDisP(dataMat[j,:].T,gaussMean[i],\
                      gaussCor[i])) for i in range(k)]
        funVal *= sum(likehoodVal)
    return math.log(funVal)

def gaussClu(dataMat,k):
    dataMat = np.mat(dataMat)
    m,n = np.shape(dataMat)
    alpha = k*[1/k]
#    gaussMean =k*[np.mat(np.zeros((n,1)))]
#    gaussMean = [dataMat[5,:].T,dataMat[21].T,dataMat[26].T]
    gaussCor = [] #初始化多维的高斯分布
    gaussMean = []  ###利用均值聚类来初始化高斯混合聚类的系数
    gaussMeanMat, cluClass = biKmeans(dataMat,k)
    for i in range(k):
        gaussMean.append(gaussMeanMat[i,:].T)
        gaussCor.append(np.cov(dataMat[(cluClass[:,0]==0).T.tolist()[0]].T))
    oldLikeVal = likeFunVal(dataMat,gaussMean,gaussCor,alpha,k)
    numIter = 0
    while numIter <100:
        gaussMean,gaussCor,alpha = likehoodEst(dataMat,gaussMean,gaussCor,alpha,k)
        newLikeVal = likeFunVal(dataMat,gaussMean,gaussCor,alpha,k)
        if abs(newLikeVal-oldLikeVal)<1e-5:
            break
        oldLikeVal = newLikeVal
        numIter += 1
    print(numIter)    
    return gaussMean,gaussCor,alpha


def dataClu(dataMat,gaussMean,gaussCor,alpha,k):
    dataMat = np.mat(dataMat)
    m = len(dataMat)
    classList = []
    gamma = gammaVal(dataMat,gaussMean,gaussCor,alpha,k)
    for i in range(m):
        classIndex = -1; minGammIJ = -math.inf
        for j in range(k):
            if gamma[i,j] > minGammIJ:
                classIndex = j
                minGammIJ = gamma[i,j]
        classList.append(classIndex)
    return classList
#            
       
def gaussPlot(dataMat,classList,gaussMean,k):
    dataMat = np.mat(dataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].T.tolist()[0],dataMat[:,1].T.tolist()[0],\
              c =list((np.array(classList)+1)*30))
    ax.scatter([float(gaussMean[t][0]) for t in range(k)],[float(gaussMean[t][1]) for t in range(k)],\
                marker = 'x',c =[30,60,90,120])

    plt.show()             
            
            
dataMat = loadDataSet('testSet.txt')       
k=4

clusterCen, cluClass = biKmeans(dataMat,k)         
plotData(dataMat,clusterCen,cluClass,k)

   
gaussMean,gaussCor,alpha = gaussClu(dataMat,k)
classList = dataClu(dataMat,gaussMean,gaussCor,alpha,k)
gaussPlot(dataMat,classList,gaussMean,k)           
            



            
            
            
            