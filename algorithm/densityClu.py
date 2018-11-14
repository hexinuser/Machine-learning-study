# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:43:59 2018

@author: hexin
"""
import numpy as np
import matplotlib.pyplot as plt 


def loadDataSet(filename):
    fr = open(filename)
    dataMat = []
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataMat.append(list(map(float,line)))
    return dataMat



"""
密度聚类，调节参数使得得为被分类的点最少
DBSCAN算法：Density-Based Spatial Clustering of Applicatios with Noise
"""
def distAB(vec1,vec2):
    ##返回两个向量的距离,(行矩阵)
    return float(np.sqrt((vec1-vec2)*(vec1-vec2).T))

def epslionMin(dataMat,epsilon,minPts):
    m = np.shape(dataMat)[0]
    epsMinI = [] ; epsMinPt = []
    for i in range(m):
        epsSetI = []
        for j in range(m):
            if distAB(dataMat[i],dataMat[j]) < epsilon:
                epsSetI.append(j)
        if len(epsSetI) >= minPts:
            epsMinI.append(i)
            epsMinPt.append(epsSetI)
    return epsMinI,epsMinPt


#dataMat  = loadDataSet('123.txt')
#dataMat = np.mat(dataMat)
#epsMinI,epsMinPt = epslionMin(dataMat,0.11,5)

#    
def densityClu(dataMat,epsilon,minPts):
    dataMat = np.mat(dataMat)
    m = np.shape(dataMat)[0]
    totalDataSet = list(range(m))
    epsMinI,epsMinPt = epslionMin(dataMat,epsilon,minPts)
    k = 0; denClu = []; omega = epsMinI[:]
    while len(omega) > 0:
        totalDataSetOld = totalDataSet[:]
        t = np.random.randint(0,len(omega))
#        dataIMin = epsMinI[t]
        epsMinPtI = epsMinPt[epsMinI.index(omega[t])]
        totalDataSet = list(set(totalDataSet).difference(set(epsMinPtI)))
        while len(epsMinPtI) > 0:
            xi = epsMinPtI[0]
            epsMinPtI.remove(xi)
            if xi in epsMinI:
                xiIndex = epsMinI.index(xi)
                delta = list(set(epsMinPt[xiIndex]).intersection(set(totalDataSet)))
                epsMinPtI += delta
                totalDataSet = list(set(totalDataSet).difference(set(delta)))
        denClu.append(list(set(totalDataSetOld).difference(set(totalDataSet))))
        omega = list(set(omega).difference(set(denClu[-1])))
        k += 1
    return denClu



def densityPlot(dataMat,denClu):
    dataMat = np.mat(dataMat)
    m = np.shape(dataMat)[0]
    classList = np.array(np.zeros(m))-1
    n = len(denClu)
    for i in range(n):
        classList[denClu[i]] = (i+2)*int(250/(n+2))
        classList[classList<0] = 255  
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].T.tolist()[0],dataMat[:,1].T.tolist()[0],\
              c =list((classList+1)*int(250/(n+1))))

    plt.show()     


                
#dataMat  = loadDataSet('testSet.txt')           
#denClu = densityClu(dataMat,1.6,4)
#densityPlot(dataMat,denClu)
#


"""
层次聚类：AGNES算法 (AGglomerative NESting)
"""
def distCluMin(vecList1,vecList2):
    #向量列表的元素的行矩阵
    minDist = np.inf
    for i in range(np.shape(vecList1)[0]):
        for j in range(np.shape(vecList2)[0]):
            dist12 = distAB(vecList1[i],vecList2[j])
            if dist12 < minDist:
                minDist = dist12
    return minDist

def distCluMax(vecList1,vecList2):
    #向量列表的元素的行矩阵
    maxDist = -np.inf
    for i in range(np.shape(vecList1)[0]):
        for j in range(np.shape(vecList2)[0]):
            dist12 = distAB(vecList1[i],vecList2[j])
            if dist12 > maxDist:
                maxDist = dist12
    return maxDist

def distCluAvg(vecList1,vecList2):
    distSum = 0
    for i in range(np.shape(vecList1)[0]):
        for j in range(np.shape(vecList2)[0]):
            distSum += distAB(vecList1[i],vecList2[j])
    avgDist = distSum/(np.shape(vecList1)[0]+np.shape(vecList2)[0])
    return avgDist
"""
上述三种来表示不同簇类的距离定义，以此可以决定不同的层次聚类,相对来说选择均值或极大较多
"""

def findMinMatIndex(M):
    m = np.shape(M)[0]
    minValue = np.inf 
    for i in range(m):
        for j in range(i+1,m):
            if M[i,j] < minValue:
                minValue = M[i,j]
                minI = i; minJ= j
    return minI, minJ 

def hierClu(dataMat,k,distMethon = distCluMax):
    dataMat = np.mat(dataMat)
    m = np.shape(dataMat)[0]
#    classClu = [dataMat[i] for i in range(m)] #初始化每个元素为一个簇
    classClu = [[i] for i in range(m)]
    M = np.mat(np.zeros((m,m)))
    for i in range(m):
        for j in range(i+1,m):
#            M[i,j] = distCluMin(classClu[i],classClu[j])
            M[i,j] = distMethon(dataMat[classClu[i]],dataMat[classClu[j]])
            M[j,i] = M[i,j]
    q = m
    while q > k:
        i,j = findMinMatIndex(M) #反回矩阵最小值的索引（两个数组对应最小值的行列下标）
        classClu[i] += classClu[j]
        if j != q:
            for t in range(j+1,q):
                classClu[t-1] = classClu[t]
        classClu.pop()
        M = np.delete(M,j,axis=0); M = np.delete(M,j,axis=1)
        for t in range(q-1):
            M[i,t] = distMethon(dataMat[classClu[i]],dataMat[classClu[t]])
            M[t,i] = M[i,t]
        q = q- 1
    return classClu


def hierPlot(dataMat,classClu,k):
    dataMat = np.mat(dataMat)
    m = np.shape(dataMat)[0]
    classList = np.array(np.zeros(m))-1
    for i in range(k):
        classList[classClu[i]] = (i+2)*int(250/(k+2))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].T.tolist()[0],dataMat[:,1].T.tolist()[0],\
              c =list(classList))

    plt.show()   
k = 6
dataMat  = loadDataSet('testSet2.txt')           
classClu = hierClu(dataMat,k)
classClu1 = hierClu(dataMat,k,distMethon = distCluMin)
classClu2 = hierClu(dataMat,k,distMethon = distCluAvg)
hierPlot(dataMat,classClu,k)
hierPlot(dataMat,classClu1,k)
hierPlot(dataMat,classClu2,k)















