# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 13:48:43 2018

@author: hexin
"""

import dTrees
import plotTree

import numpy as np

def watermelonTree():
    #读取西瓜数据
    watermelon2 = open('watermelon2.0.txt')
    #文档的第一列为编号，无需属性判断,需删除, 第一行为样本属性需单独存储标签并删除，最后一列为分类结果
    dataWL2 = [data.strip().split(',') for data in watermelon2 ] #将数据存储为列表
    watermelonLable = dataWL2.pop(0)[1:-1]  
    dataWL2 = np.delete(np.array(dataWL2),0,1)
    labelValClass =dTrees.classProp(dataWL2,watermelonLable)  
    watermelonTree = dTrees.creatTrees(dataWL2,watermelonLable,labelValClass,'ID3')
    plotTree.createPlot(watermelonTree)
    

def testLenses():
    lensesTxt = open('lenses.txt')
    lenses = np.array([lense.strip().split('\t') for lense in lensesTxt ]) #将数据存储为列表
    lenseLable = ['age','prescript','astigamtic','tearRate']
    lenseTree = dTrees.creatTrees(lenses, lenseLable,'C4.5')
    plotTree.createPlot(lenseTree)
    
def testWMTdata():
    #读取西瓜数据,分为训练集和测试集
    watermelon2 = open('watermelon2.0.txt')
    #文档的第一列为编号，无需属性判断,需删除, 第一行为样本属性需单独存储标签并删除，最后一列为分类结果
    dataWL2 = [data.strip().split(',') for data in watermelon2 ] #将数据存储为列表
    watermelonLable = dataWL2.pop(0)[1:-1]  
    dataWL2 = np.delete(np.array(dataWL2),0,1)
    m,n = dataWL2.shape
    trainIndex=[0,1,2,5,6,9,13,14,15,16]
    trainSet = dataWL2[trainIndex,:]
    testIndex = list(range(len(dataWL2)))
    for i in trainIndex:
        testIndex.remove(i)
    testSet = dataWL2[testIndex,:]
    return trainSet,testSet,watermelonLable
        
def testWMtree():
    trainSet,testSet,watermelonLable = testWMTdata()
    labelValClass =dTrees.classProp(trainSet,watermelonLable)  
    watermelonTree = dTrees.creatTrees(trainSet,watermelonLable,labelValClass,'ID3')
    plotTree.createPlot(watermelonTree)
    rightclass0 = 0
    for i in range(len(trainSet)):
        testdata = trainSet[i,:]
        classdata = dTrees.classify(watermelonTree,watermelonLable,testdata[:-1])
        if classdata == testdata[-1]:
            rightclass0 +=1
    rightProb0 = rightclass0/len(trainSet)
    rightclass = 0
    for i in range(len(testSet)):
        testdata = testSet[i,:]
        classdata = dTrees.classify(watermelonTree,watermelonLable,testdata[:-1])
        if classdata == testdata[-1]:
            rightclass +=1
    rightProb = rightclass/len(testSet)
    print(rightProb0); print(rightProb)


#trainSet,testSet,watermelonLable = testWMTdata()
#labelValClass =dTrees.classProp(trainSet,watermelonLable)  
#watermelonTree = dTrees.creatTrees(trainSet,watermelonLable,labelValClass,'ID3')
testWMtree()
#testLenses()














