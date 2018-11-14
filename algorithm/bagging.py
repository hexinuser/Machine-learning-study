# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:31:59 2018

@author: hexin
"""

import numpy as np
import matplotlib.pyplot as plt 
from pylab import mpl 
mpl.rcParams['font.sans-serif'] = ['SimHei']  


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
                    minError = testError
                    bestTree['natI'] = i; bestTree['val'] = testValI; bestTree['ineq'] = inequality
    return bestTree
#, minError,bestLabel



def bagging(dataMat,labelMat,T=60):
    m = len(dataMat); sigTrees = []
    for i in range(T):
        dataMatI = []; labelMatI = []
        for j in range(m):
            t = np.random.randint(m)                
            dataMatI.append(dataMat[t])
            labelMatI.append(labelMat[t])
        dataMatI = np.mat(dataMatI); labelMatI = np.mat(labelMatI).T
        D = np.mat(np.ones((m,1))/m)
        sigTrees.append(sigTree(dataMatI,labelMatI,D))   
    return sigTrees
        
                
    


def testVecLabel(testVec,baggingTrees):
    testClass = [0,0]
    for i in range(len(baggingTrees)):
        if baggingTrees[i]['ineq'] == 'lt':
            if testVec[baggingTrees[i]['natI']] <= baggingTrees[i]['val']:     
                testLabel  = -1
            else:
                testLabel = 1
        else:
            if testVec[baggingTrees[i]['natI']] > baggingTrees[i]['val']:
                testLabel  = -1
            else:
                testLabel = 1
        if testLabel == -1:
            testClass[0] += 1
        else:
            testClass[1] += 1
    if testClass[0]>testClass[1]:
        testClassVal = -1
    elif testClass[0]<testClass[1]:
        testClassVal = 1
    else:
        testClassVal = (np.random.randint(2)-0.5)*2
    return testClassVal

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
    baggingTrees = bagging(trainData,trainLabel)
    m = np.shape(trainData)[0]
    trainErrorCount = 0
    for i in range(m):
        testClass = testVecLabel(trainData[i],baggingTrees)
        if testClass != trainLabel[i]:
            trainErrorCount += 1 
    print('the error rate of trainSet is %.5f'%(trainErrorCount/m))
    testDate,testLabel = loadSet('horseColicTest2.txt')
    m1 = np.shape(testDate)[0]
    trainErrorCount1 = 0
    for i in range(m1):
        testClass = testVecLabel(testDate[i],baggingTrees)
        if testClass != testLabel[i]:
            trainErrorCount1 += 1 
    print('the error rate of testSet is %.5f'%(trainErrorCount1/m1))

testBoosting()
