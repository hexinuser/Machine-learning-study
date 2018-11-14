# -*- coding: utf-8 -*-
"""
input: dataSet: M*N 每一行就是一个样品，每个样品的N个元素为对应其属性，最后一个元素
                为对应的预测判断属性
       labels : 每个样品对应属性的标签

Created on Wed Apr 18 17:45:29 2018

@author: hexin
"""

import math
import numpy as np
import operator


def createDataSet():
    #测试数据，验证信息熵的计算
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'maybe'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    dataSet = np.array(dataSet)
    #change to discrete values
    return dataSet, labels
dataSet, labels = createDataSet()
#dataEnt = calcuInfoEnt(dataSet)


def calcuInfoEnt(dataSet):
    #计算数据的信息熵,数据可由字典或数组存贮，每个样本就是一个数组或字典
    numDate = len(dataSet) #数据样本的总个数
    dataEnt = 0 #信息熵初始化
    classCount = {} #数据包含分类标记，在最后一列
    for dataSamp in dataSet:  #每一行遍历，对每个样本
        classCount[dataSamp[-1]] = classCount.get(dataSamp[-1],0) + 1 #字典键对应值不存在，初始化为0 在计算
    for key in classCount.keys():
        probClassI = classCount[key]/numDate
        dataEnt -= probClassI*math.log(probClassI,2)
    return dataEnt


def classifyDate0(dataSet,k):
    #按第k个属性值对dataSet进行分类，不同属性值对应一个子数据集
    dataSetK = dataSet[:,k]
    classifyIndex={} #存贮字典，键为k个属性的属性值，对应为值为属于该属性值的行索引
    n = len(dataSetK)  #总样本个数
    for j in range(n):     
        if dataSetK[j] not in classifyIndex.keys():
            classifyIndex[dataSetK[j]]=[j]
        else:
            classifyIndex[dataSetK[j]].append(j)
    classifyDate = {}  #根据行索引得到对应分组的数据集
    for key in classifyIndex.keys():
        classifyDate[key] = dataSet[classifyIndex[key],:]
    return classifyDate
            
#classifyDate = classifyDate0(dataSet,0)
    

def calcuInfoGain(dataSet):
    ##ID3决策树选择, 对可取值数目较多的属性有所偏好
    ##计算每个属性的信息增益，返回对应最大的属性索引
    dataEnt = calcuInfoEnt(dataSet)
    m,n = dataSet.shape 
    for i in range(n-1):
        InfoGain = dataEnt
        classifyDate =classifyDate0(dataSet,i) #得到按第i个属性值分类的数据集字典
        for key in classifyDate.keys():
            m0 = len(classifyDate[key])
            InfoGain -= (m0/m)*calcuInfoEnt(classifyDate[key])
        if i == 0:
            InfoGain0 = InfoGain
            bestInfoGrinIndex = 0  #初始化最大的信息增益的值
        else:
            if InfoGain > InfoGain0:
                bestInfoGrinIndex = i
                InfoGain0 = InfoGain
    return bestInfoGrinIndex
           
#aa=calcuInfoGain(dataSet)      
    

def calcuGrinRatio(dataSet):
    ##C4.5决策树算法,对可取值数目较小有所偏好，找出信息增益高出平均水平的属性，选择增益率最高
    #计算每每个属性的信息增益率，和只利用信息增益计算进行比较,返回最大信息增益的属性
    dataEnt = calcuInfoEnt(dataSet) #先计算信息熵
    m,n = dataSet.shape 
    #循环计算每个属性的信息熵
    for i in range(n-1):
        GrinRation = dataEnt
        InstantV = 0 #求信息增益率的固定值
        classifyDate =classifyDate0(dataSet,i)
        for key in classifyDate.keys():
            m0 = len(classifyDate[key])
            t = m0/m
            InstantV -= t*math.log(t,2)
            GrinRation -= t*calcuInfoEnt(classifyDate[key])
        if InstantV==0:
             bestGrinRationIndex = i
             break
        else:
            if i == 0:
                GrinRation0 = GrinRation/(InstantV+1e-20)#加一个足够小量，防止分母为0 
                bestGrinRationIndex = 0
            if i>0:
                GrinRation1 = GrinRation/(InstantV+1e-20)
                if GrinRation1 > GrinRation0:
                    bestGrinRationIndex = i
                    GrinRation0 = GrinRation1
    return bestGrinRationIndex
#aa=calcuInfoGain(dataSet)   
    

def calcuGini(dataSet):
    ##CART决策树，计数基尼值
    m = len(dataSet) #数据样本的总个数
    gini = 1 #数据集的基尼值
    classCount = {} #数据包含分类标记，在最后一列
    for dataSamp in dataSet:  #每一行遍历，对每个样本
        classCount[dataSamp[-1]] = classCount.get(dataSamp[-1],0) + 1 #字典键对应值不存在，初始化为0 在计算
    for key in classCount.keys():
        probClassI = classCount[key]/m
        gini -= probClassI**2
    return gini
    
def calcuGiniIndex(dataSet):
    ##CART决策树，基于基尼指数选择分类属性
    ##求取每个属性的基尼系数
    m,n = dataSet.shape
    giniIndex = 0
    for i in range(n-1):
        classDataSet = classifyDate0(dataSet,i)
        for key in classDataSet.keys():
            giniK = calcuGini(classDataSet[key])
            numClass = len(classDataSet[key])
            giniIndex += numClass/m*giniK
        if i == 0:
            minGiniIndex0 =  giniIndex
            bestGiniIndex = 0
        else:
            if giniIndex < minGiniIndex0:
                bestGiniIndex = i
                minGiniIndex0 =  giniIndex
    return bestGiniIndex


def majorClass(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount += 1
    sortedClassCount =sorted(classCount.items(), key=operator.itemgetter(1),reverse=True) #排序得到的是元组列表
    return sortedClassCount[0][0]

def classProp(dataSet,labels):
    #返回第k个属性的每个属性值的最多分类
    labelValClass = {}
    for k in range(len(labels)):
        proValueClass = {}
        dataMatK = dataSet[:,k]
        dataMatKUni = list(set(dataMatK))
        classList = list(set(dataSet[:,-1]))
        for i in range(np.shape(dataSet)[0]):
            if dataMatK[i] not in proValueClass:
                proValueClass[dataMatK[i]] = np.zeros(len(classList))
                t = classList.index(dataSet[i,-1])
                proValueClass[dataMatK[i]][t] = 1
            else:
                t = classList.index(dataSet[i,-1])
                proValueClass[dataMatK[i]][t] += 1
        labelKclass = []
        for key in dataMatKUni:
             labelKclass.append(classList[proValueClass[key].argmax()])
        labelValClass[labels[k]] = (dataMatKUni,labelKclass)
    return labelValClass



def creatTrees(dataSet,labelss,labelValClass,method = 'ID3'):
    ##决策树的构建选取方法，默认采取ID3决策树以信息增益构建 
    labels = labelss[:]
    classlist = [example[-1] for example in dataSet]  #分类列表
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataSet[0]) == 1:
        return majorClass(classlist) #当属性分解结束，只剩下类列，但对应分组的类别不一定唯一，我们返回类最多  
    if method == 'ID3':
        bestGrinRationIndex = calcuInfoGain(dataSet) ##信息增益的决策树
    else:
        if method == 'C4.5':  
            bestGrinRationIndex = calcuGrinRatio(dataSet)  #C4.5基于信息增益率的决策树
        else:
            if method == 'CART':
                bestGrinRationIndex = calcuGiniIndex(dataSet)  #基于基尼指数的CART决策树
            else:
                print('please choose a right method!!')
    bestLabel = labels[bestGrinRationIndex]
    bestLabelAllValues = labelValClass[bestLabel]
    myTree = {bestLabel:{}}  #以字典嵌套字典的方式返回决策树
    bestLabelValues = set([example[bestGrinRationIndex] for example in dataSet]) #当前删除属性的所有可能值的集合
    del (labels[bestGrinRationIndex]) #删除已划分的属性标签    
    
    for i in range(len(bestLabelAllValues[0])):
        value =bestLabelAllValues[0][i]
        if value in bestLabelValues:        
            subLabels = labels[:]
            subDataSet = classifyDate0(dataSet,bestGrinRationIndex)[value]
            m,n = subDataSet.shape
            realSubSet = []
            for i in range(n):
                if i<bestGrinRationIndex:
                    realSubSet.append(subDataSet[:,i])
                elif i>bestGrinRationIndex:
                    realSubSet.append(subDataSet[:,i])
            realSubSet=np.array(realSubSet).T
            myTree[bestLabel][value] = creatTrees(realSubSet,subLabels,labelValClass,method)
        else:
            myTree[bestLabel][value] = bestLabelAllValues[1][i]
    return myTree


#def creatTrees(dataSet,labels,method = 'ID3'):
#    ##决策树的构建选取方法，默认采取ID3决策树以信息增益构建 
#    classlist = [example[-1] for example in dataSet]  #分类列表
#    if classlist.count(classlist[0]) == len(classlist):
#        return classlist[0]
#    if len(dataSet[0]) == 1:
#        return majorClass(classlist) #当属性分解结束，只剩下类列，但对应分组的类别不一定唯一，我们返回类最多  
#    if method == 'ID3':
#        bestGrinRationIndex = calcuInfoGain(dataSet) ##信息增益的决策树
#    else:
#        if method == 'C4.5':  
#            bestGrinRationIndex = calcuGrinRatio(dataSet)  #C4.5基于信息增益率的决策树
#        else:
#            if method == 'CART':
#                bestGrinRationIndex = calcuGiniIndex(dataSet)  #基于基尼指数的CART决策树
#            else:
#                print('please choose a right method!!')
#    bestLabel = labels[bestGrinRationIndex]
#    myTree = {bestLabel:{}}  #以字典嵌套字典的方式返回决策树
#    bestLabelValues = set([example[bestGrinRationIndex] for example in dataSet]) #当前删除属性的所有可能值的集合
#    del (labels[bestGrinRationIndex]) #删除已划分的属性标签
#    for value in bestLabelValues:
#        subLabels = labels[:]
#        subDataSet = classifyDate0(dataSet,bestGrinRationIndex)[value]
#        m,n = subDataSet.shape
#        realSubSet = []
#        for i in range(n):
#            if i<bestGrinRationIndex:
#                realSubSet.append(subDataSet[:,i])
#            elif i>bestGrinRationIndex:
#                realSubSet.append(subDataSet[:,i])
#        realSubSet=np.array(realSubSet).T
#        myTree[bestLabel][value] = creatTrees(realSubSet,subLabels,method)
#    return myTree
#
    

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  #得到第一个索引在标签中的索引
    key = testVec[featIndex] ##得到第对应属性的属性值，判定，下一层字典
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


### 存储和读取决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb+')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
   
labelValClass = classProp(dataSet,labels)    
#mytree =  creatTrees(dataSet,labels,labelValClass)
#mytree =  creatTrees(dataSet,labels)
#end=classify(mytree,['no surfacing','flippers'],['1','0'])    

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    