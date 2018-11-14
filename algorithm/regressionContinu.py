# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:13:49 2018

@author: hexin
"""

import numpy as np
import matplotlib.pyplot as plt
import urllib3
from time import sleep


def loadDataSet(fileneme):
    dataMat = []; valueSet = []
    fr = open(fileneme)
    for line in fr.readlines():
        line = list(map(lambda x: float(x),line.strip().split('\t'))) 
        dataMat.append(line[:-1])
        valueSet.append(line[-1])
    return dataMat,valueSet
#a,b=loadDataSet('ex0.txt')

def standRegr(dataMat,valueMat):
    #仅当数据集是行满秩的时候才能求到最优曲线,lam为岭回归系数，修正X^T*X不正定的情况，默认不修正
    #其最优w是利用线性均方误差来求解(无偏估计)的，如果数据量较小可能大不到行满秩要求
    #容易出现欠拟合的现象
    #最小化lam系数拟合
    dataMat = np.mat(dataMat); valueMat = np.mat(valueMat).T
    if np.linalg.det(dataMat.T*dataMat) != 0:
        w = np.linalg.inv(dataMat.T*dataMat)*dataMat.T*valueMat
    else:
        print('数据集的矩阵秩不是行满秩')
    return w.A.flatten().tolist()   #返回列表存储


def plotLine(dataMat,valueSet,w):
    X = list(map(lambda x: x[1],dataMat))
    Y = valueSet
    x = np.arange(min(X),max(X),0.01)
    y = w[0]+w[1]*x
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X,Y,c='b')
    ax.plot(x,y,c='red')
    plt.show()
    

def localWSR(dataMat,valueMat, prediData,k):
    ##局部加权线性回归，对待预测点附近的点赋予更高权重，来拟合
    ##不同的预测点prediData会得到不同的拟合曲线
    ##k为高斯核的所需值,k越小,预测点附近的权重越大,用于回归的点越小
    dataMat = np.mat(dataMat); valueMat = np.mat(valueMat).T
    m,n = np.shape(dataMat)
    wMat = np.eye(m) #每个数据的权重
    prediData = np.mat(prediData).T

    for i in range(m):
        wMat[i,i] = np.exp(-(prediData-dataMat[i].T).T*(prediData-dataMat[i].T)/(2*k**2))
        #不同的初始值赋予不同点不同的权重，此处为高斯核，也有其他的数据
    if np.linalg.det(dataMat.T*wMat*dataMat) != 0:
        w = np.linalg.inv(dataMat.T*wMat*dataMat)*dataMat.T*wMat*valueMat
    else:
        print('数据集的矩阵秩不是行满秩')
        return
    return w.A.flatten().tolist()   #返回列表存储

def prediArr(prediVec,dataMat,valueSet,k):
    #对多个预测点循环预测回归系数
    wList = []
    for prePoint in prediVec:
        wList.append(localWSR(dataMat,valueSet, prePoint,k))
    return wList

def plotWSR(dataMat,valueSet,k):
    X = list(map(lambda x: x[1],dataMat))
    Y = valueSet
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X,Y,c='b',s=2)
    x = sorted(X) 
    testList = list(map(lambda t: [1,t],x))
    W = prediArr(testList,dataMat,valueSet,k)
    y = list(map(lambda a,b: a*b[1]+b[0], x,W))
    ax.plot(x,y,c='red')
    plt.show()
    
#dataMat,labelMat = loadDataSet('ex0.txt') 
#w = standRegr(dataMat,valueSet)
##改数据的第一列全部初始化为1,类似二维平面上找最佳拟合曲线
#plotLine(dataMat,valueSet,w)
##判断两个变量的线性相关矩阵，得到不同变量的线性相关性，判断是否可利用线性拟合
#corrXY = np.corrcoef(np.array(dataMat)[:,1],np.array(valueSet))
#
#plotWSR(dataMat,valueSet,1)   #k=1是等价于各个点的权重相同
#plotWSR(dataMat,valueSet,0.05) ##可以得到不同的权重会导致不同的拟合曲线
#plotWSR(dataMat,valueSet,0.01) 
#plotWSR(dataMat,valueSet,0.003)  ##数值越小，考虑的噪声影响越小，容易出现过拟合现象
        
#t=1  #比较不同的模型数值
#dataMat1,valueSet1= loadDataSet('abalone.txt')        
#w1 = prediArr(dataMat1[0:99], dataMat1[0:99], valueSet1[0:99],t)
#errorStat = 0
#ErrorValue = [(float(np.mat(w1[i])*np.mat(dataMat1[i]).T)-valueSet1[i])**2 for i in range(99)]
#
#errorSq = sum(ErrorValue)
#print('the trainSet square error  is %.5f'%(errorSq))
#
#w2 = prediArr(dataMat1[100:199], dataMat1[0:99], valueSet1[0:99],t)
#errorStat = 0
#ErrorValue1 = [(float(np.mat(w2[i])*np.mat(dataMat1[i+100]).T)-valueSet1[i+100])**2 for i in range(99)]      
#errorSq1 = sum(ErrorValue1)
#print('the trainSet square error  is %.5f'%(errorSq1))     
    
    
def standRegrCor(dataMat,valueMat,lam = 0.0):
    #仅当数据集是行满秩的时候才能求到最优曲线,lam为岭回归系数，修正X^T*X不正定的情况，默认不修正
    #其最优w是利用线性均方误差来求解(无偏估计)的，如果数据量较小可能大不到行满秩要求
    #容易出现欠拟合的现象 #最小化lam系数拟合
    w = np.linalg.inv(dataMat.T*dataMat+lam*np.mat(np.eye(np.shape(dataMat)[1])))*dataMat.T*valueMat
    return w.A.flatten().tolist()   #返回列表存储

#dataMat,labelMat= loadDataSet('abalone.txt')
#
#def normData(dataMat,labelMat):
#    dataMat = np.mat(dataMat); labelMat = np.mat(labelMat).T
#    dataMean = np.mean(dataMat,0); dataVar  = np.var(dataMat,0)
#    labelMean = np.mean(labelMat,0)
#    dataMat = (dataMat - dataMean)/dataVar; labelMat = labelMat -labelMean
#    numIter = 30
#    wMat = []
#    for i in range(numIter):
#        ws = standRegrCor(dataMat,labelMat,np.exp(i-10))
#        wMat.append(ws)
#    return wMat
##交叉验证测试左右的lambda值
#w = normData(dataMat,labelMat)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(ww)
#plt.show()


#def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
#    sleep(10)
#    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
#    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
#    pg = urllib3.urlopen(searchURL)
#    retDict = json.loads(pg.read())
#    for i in range(len(retDict['items'])):
#        try:
#            currItem = retDict['items'][i]
#            if currItem['product']['condition'] == 'new':
#                newFlag = 1
#            else: newFlag = 0
#            listOfInv = currItem['product']['inventories']
#            for item in listOfInv:
#                sellingPrice = item['price']
#                if  sellingPrice > origPrc * 0.5:
#                    print( "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
#                    retX.append([yr, numPce, newFlag, origPrc])
#                    retY.append(sellingPrice)
#        except: print 'problem with item %d' % i
#    
#def setDataCollect(retX, retY):
#    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
#    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
#    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
#    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
#    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
#    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
#    
#def crossValidation(xArr,yArr,numVal=10):
#    m = len(yArr)                           
#    indexList = range(m)
#    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows
#    for i in range(numVal):
#        trainX=[]; trainY=[]
#        testX = []; testY = []
#        random.shuffle(indexList)
#        for j in range(m):#create training set based on first 90% of values in indexList
#            if j < m*0.9: 
#                trainX.append(xArr[indexList[j]])
#                trainY.append(yArr[indexList[j]])
#            else:
#                testX.append(xArr[indexList[j]])
#                testY.append(yArr[indexList[j]])
#        wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
#        for k in range(30):#loop over all of the ridge estimates
#            matTestX = mat(testX); matTrainX=mat(trainX)
#            meanTrain = mean(matTrainX,0)
#            varTrain = var(matTrainX,0)
#            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
#            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
#            errorMat[i,k]=rssError(yEst.T.A,array(testY))
#            #print errorMat[i,k]
#    meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
#    minMean = float(min(meanErrors))
#    bestWeights = wMat[nonzero(meanErrors==minMean)]
#    #can unregularize to get model
#    #when we regularized we wrote Xreg = (x-meanX)/var(x)
#    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
#    xMat = mat(xArr); yMat=mat(yArr).T
#    meanX = mean(xMat,0); varX = var(xMat,0)
#    unReg = bestWeights/varX
#    print "the best model from Ridge Regression is:\n",unReg
#    print "with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat)

        
        
        
        
        
        
        
        