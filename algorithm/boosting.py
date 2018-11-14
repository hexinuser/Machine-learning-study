# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:38:19 2018

@author: hexin
"""
import numpy as np
import matplotlib.pyplot as plt 
from pylab import mpl 
mpl.rcParams['font.sans-serif'] = ['SimHei']  
#解决绘图中文乱码的问题

def loadSimpData():
    datMat = np.array([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return np.mat(datMat),np.mat(classLabels).T

def testLabel(dataMat,labelMat,nature,testVal,inequality):
    labelArr = np.ones_like(labelMat)
    if inequality == 'lt':
        labelArr[dataMat[:,nature] <= testVal] = -1.0
    else:
        labelArr[dataMat[:,nature] > testVal] = -1.0
    return labelArr


def sigTree(dataMat,labelMat,D,numDiv=10):
    #D表示每个数据的权重，为一个m*1的矩阵
    m,n = np.shape(dataMat)  #m个数据，n个属性
    minError = np.Inf; bestTree = {}
    for i in range(n): ##迭代属性
        minValI = np.min(dataMat[:,i]); maxValI = np.max(dataMat[:,i])
        stepI = (maxValI-minValI)/numDiv
        for j in range(-1,numDiv+1):
            testValI = minValI + j*stepI
            for inequality in ['lt','rt']:
                labelArr = testLabel(dataMat,labelMat,i,testValI,inequality)
                errorStat = np.zeros_like(labelMat)
                errorStat[labelMat!=labelArr] = 1
                testError = D.T*errorStat
                if testError < minError:
                    bestLabel = labelArr.copy()
                    minError = testError
                    bestTree['natI'] = i; bestTree['val'] = testValI; bestTree['ineq'] = inequality
    return bestTree, minError,bestLabel
#dataMat, labelMat = loadSimpData()
#D = np.mat(np.ones((5,1))/5)
#a,b,c = sigTree(dataMat,labelMat,D)
#print(a);print(b);print(c)


def boosting(dataMat,labelMat,T=40):
    #T表示训练轮数
    boostTree= [] ; boostAlpha = []
    dataMat = np.mat(dataMat); labelMat = np.mat(labelMat).T
    m,n = np.shape(dataMat)
    D = np.mat(np.ones((m,1))/m)
    for t in range(T):
        interTree, errorRate,bestLabel = sigTree(dataMat,labelMat,D)
        if errorRate>0.5:
            print('this method is not suit for this data,please change other one')
            break
        else:
            alpha = float(0.5*np.log((1-errorRate)/errorRate))
            boostAlpha.append(alpha)
        for i in range(m):
            if labelMat[i] == bestLabel[i]:
                D[i] *= np.exp(-alpha)
            else:    
                D[i] *= np.exp(alpha)
        D = D/D.sum()
        boostTree.append(interTree)
    return boostTree,boostAlpha

def testVecLabel(testVec,boostTree,boostAlpha):
    hx = 0
    for i in range(len(boostTree)):
        if boostTree[i]['ineq'] == 'lt':
            if testVec[boostTree[i]['natI']] <= boostTree[i]['val']:     
                testLabel  = -1.0
            else:
                testLabel = 1
        else:
            if testVec[boostTree[i]['natI']] > boostTree[i]['val']:
                testLabel  = -1.0
            else:
                testLabel = 1
        hx += boostAlpha[i]*testLabel
    if np.sign(hx)> 0 :
        testClass = 1
    else:
        testClass = -1
    return testClass

def loadSet(filename):
    fr = open(filename)
    dataMat = []; labelMat = []
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataMat.append(list(map(lambda x: float(x),line[:-1])))
        labelMat.append(int(float(line[-1])))
    return dataMat,labelMat


def testBoosting():
    trainData,trainLabel = loadSet('horseColicTraining2.txt')
    boostTree,boostAlpha = boosting(trainData,trainLabel)
    m = np.shape(trainData)[0]
    trainErrorCount = 0
    for i in range(m):
        testClass = testVecLabel(trainData[i],boostTree,boostAlpha)
        if testClass != trainLabel[i]:
            trainErrorCount += 1 
    print('the error rate of trainSet is %.5f'%(trainErrorCount/m))
    testDate,testLabel = loadSet('horseColicTest2.txt')
    m1 = np.shape(testDate)[0]
    trainErrorCount1 = 0
    for i in range(m1):
        testClass = testVecLabel(testDate[i],boostTree,boostAlpha)
        if testClass != testLabel[i]:
            trainErrorCount1 += 1 
    print('the error rate of testSet is %.5f'%(trainErrorCount1/m1))

#testBoosting()


def testBoosting1(T):
    trainData,trainLabel = loadSet('horseColicTraining2.txt')
    boostTree,boostAlpha = boosting(trainData,trainLabel,T)
    m = np.shape(trainData)[0]
    trainErrorCount = 0
    for i in range(m):
        testClass = testVecLabel(trainData[i],boostTree,boostAlpha)
        if testClass != trainLabel[i]:
            trainErrorCount += 1 
            trainError = trainErrorCount/m
#    print('the error rate of trainSet is %.5f'%(trainErrorCount/m))
    testDate,testLabel = loadSet('horseColicTest2.txt')
    m1 = np.shape(testDate)[0]
    trainErrorCount1 = 0
    for i in range(m1):
        testClass = testVecLabel(testDate[i],boostTree,boostAlpha)
        if testClass != testLabel[i]:
            trainErrorCount1 += 1 
#    print('the error rate of testSet is %.5f'%(trainErrorCount1/m1))
            testError = trainErrorCount1/m1
    return trainError,testError

def plotErrorRate():
    T =list(range(100)); X1 = []; X2 = []
    for t in T:
        trainError, testError = testBoosting1(t)
        X1.append(trainError)
        X2.append(testError)
    figuer = plt.figure()
    ax1 = figuer.add_subplot(211)
    ax1.plot(T,X1)
    plt.xlabel('训练个数'); plt.ylabel('训练集错误率');
    ax2 = figuer.add_subplot(212)
    ax2.plot(T,X2)
    plt.xlabel('训练个数'); plt.ylabel('测试错误率');
    plt.show()
    
plotErrorRate()        
        
            
def plotROC():
    trainData,trainLabel = loadSet('horseColicTraining2.txt')
    boostTree,boostAlpha = boosting(trainData,trainLabel)
    m = np.shape(trainData)[0]
    testClass = []
    for i in range(m):
        testClass.append ( testVecLabel(trainData[i],boostTree,boostAlpha))
    testClass = np.array(testClass)
    sortBytestClass  = np.argsort(testClass)
    trainLabel =np.array(trainLabel)[sortBytestClass][::-1]
    FN = trainLabel[trainLabel>0].sum()
    TN = m - FN
    TP = 0; FP = 0
    X = []; Y = []
    for i in range(m):
        X.append(FP/(FP+TN)); Y.append(TP/(TP+FN))
        if trainLabel[i] == 1:
            TP += 1 ##真正例
            FN -= 1
        else:
            FP += 1 ##假正例
            TN -= 1
    figuer = plt.figure()
    ax = figuer.add_subplot(111)
    ax.plot(X,Y)
    plt.xlabel('假阳率'); plt.ylabel('真阳率');
    plt.xlim(0,1);plt.ylim(0,1)
    plt.title('ROC曲线')
    plt.show()
                
#plotROC()            
            
    
    





















            
            
                
        
                
    