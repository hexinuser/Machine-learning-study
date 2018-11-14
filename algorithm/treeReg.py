# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:10:55 2018

@author: hexin
"""
import numpy as np
import matplotlib.pyplot as plt 


def loadDataSet(fileName):      
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) 
        dataMat.append(fltLine)
    return dataMat

#a = loadDataSet('ex2.txt')
    
def binSplitDataSet(dataSet, feature, value):
    #对某属性按某个值进行二元切分
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0]]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0]]
    return mat0,mat1

def regLeaf(dataSet):
    #得到当前数据回归值的均值，
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    #得到当前数据回归值的与对应均值的差的平方和
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]



def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]  #划分的误差容许 ops可根据数据量自设
    tolN = ops[1]  #表示切分的最少样本数，当切分的样本少于tolN时，停止切分

    if len(np.unique(np.array(dataSet[:,-1]))) == 1: 
        #当前回归值相同，不需要进行划分
        return None, leafType(dataSet) #None是为了和返回值对应
    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf #容许的误差下降值
    bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):  #数据集的最后一列是回归值,故只有n-1个属性，选取最佳属性
        for splitVal in np.unique(np.array(dataSet[:,featIndex])):  #唯一的属性值集合选取最优划分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                #预剪枝，防止数据过拟合
                continue #继续最内层的循环
            newS = errType(mat0) + errType(mat1)  #得到的划分误差要减少
            if newS < bestS: 
                #得到的划分误差要减少，找到误差最少的，及每个划分的数据集回归值有着更好的相似
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS: 
        #当划分的误差和原数据误差在容许范围内,不返回划分，直接做为叶节点，回归值为当前数据集的真实值的均值
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  #exit cond 3
        #表示当前不存在最优划分，同样作为叶节点
        return None, leafType(dataSet)
    return bestIndex,bestValue  #存在最优划分，返回划分的属性及其对应属性值
##回归树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    #划分需保证dataSet为mat矩阵
    #改数的优点是对以划分的属性依旧可以继续继续划分
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: 
        return val #直接返回当前叶节点的回归值
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  


def plotData(dataMat,c='b'):
    #对二维数据点进行绘制
    m,n = np.shape(dataMat)
    if n == 3: #包含第一列为1便于计算线性回归
        X = dataMat[:,1].T.tolist()[0]
    else:
        X = dataMat[:,0].T.tolist()[0]
    Y =dataMat[:,-1].T.tolist()[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X,Y,c=c)
    plt.show()

#dataMat = np.mat(loadDataSet('ex00.txt'))
#plotData(dataMat)
#trees = createTree(dataMat)



#dataMat1 = np.mat(loadDataSet('ex0.txt'))
##plotData(dataMat1)
#trees1 = createTree(dataMat1) #第一列的数据相同不会影响划分、
#
#dataMat2 = np.mat(loadDataSet('ex2.txt'))
##plotData(dataMat2)
#trees2 = createTree(dataMat2,ops=(20,10))  
##print(trees2)
"""
对比就可以得到 不同的ops选择会产生不同的决策树，通过对数据集和测试集的时间误差
来调节系数，以达到两者的一个平衡值，这就是一种后剪枝的方法，但常常不太切合实际数据
"""

def isTree(obj):
    return type(obj).__name__ == 'dict'

def getMean(tree):
    #返回当前树的根节点剪枝后，该叶节点的值
    if isTree(tree['right']): 
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): 
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if np.shape(testData)[0] == 0: 
        return getMean(tree)  
    #当前测试集划分后为空，此时的返回为叶节点，其值为原决策树对应的子树的剪枝合并的预测值
    
    if (isTree(tree['right']) or isTree(tree['left'])):
        ##当前树的两个划分仍存在子树，对测试集进行对应划分验证误差
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): 
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): 
        tree['right'] =  prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        #都是叶节点，考虑合并，是否能减少误差,次数tree['left']与tree['right']均为回归值
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #原决策树的误差
        errorNoMerge = sum(np.power(lSet[:,-1] - tree['left'],2)) +\
            sum(np.power(rSet[:,-1] - tree['right'],2))
        
        #和并后的误差
        treeMean = (tree['left']+tree['right'])/2.0 #更换当前结点为叶节点
        errorMerge = sum(np.power(testData[:,-1] - treeMean,2))
        
        if errorMerge < errorNoMerge: 
#            global mergingCount
#            mergingCount += 1
#            print('merging')
            return treeMean
        else: 
            return tree
    else: 
        return tree
    
    
##mergingCount = 0 #统计合并次数
#trainingMat = np.mat(loadDataSet('ex2.txt'))
#trainingTrees = createTree(trainingMat,ops=(0.2,1))  
##print(trainingTrees)
#testData = np.mat(loadDataSet('ex2test.txt'))
#testTrees = prune(trainingTrees, testData)
#print(testTrees)
#print(mergingCount)  #通常前后剪枝结合
#    
def treeRegVal(preVal,tree):
    fristIndex = tree['spInd']
    if preVal[fristIndex] > tree['spVal']:
        if type(tree['left']).__name__ == 'dict':
            return(treeRegVal(preVal,tree['left']))
        else:
            return tree['left']
    elif preVal[fristIndex] <= tree['spVal']:
        if type(tree['right']).__name__ == 'dict':
            return(treeRegVal(preVal,tree['right']))
        else:
            return tree['right']
    
def comprRegData(testData,testTrees):
    plotData(testData)
    Y = []
    for i in range(np.shape(testData)[0]):
        preVal = testData[i,:-1].A.flatten()
        Y.append(treeRegVal(preVal,testTrees))
    print(np.corrcoef(np.mat(Y).T,testData[:,-1],rowvar = 0)[0,1])
    testData[:,1] = np.mat(Y).T
    plotData(testData,c='r')

#comprRegData(testData,testTrees)
  
  
def linearSolve(dataSet):  
    #对每一个数据集进行线性回归拟合
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]  #包含了线性的常数项，对应w的第一项
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)  #返回的ws为矩阵
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    #返回回归值与真实值的均方误差
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat,2))


#trainingMat = np.mat(loadDataSet('ex2.txt'))
#得到模型树，根节点值对局部的线性回归值
#trainingTrees = createTree(trainingMat,modelLeaf,modelErr,ops=(1,10)) 

 
def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(ws, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[0,1:n+1]=inDat   #原数据第一列增加为1，对应线性回归的常数
    return float(X*ws)
#modelEval=regTreeEval
def treeForeCast(tree, inData):
    if not type(tree).__name__=='dict': 
        return tree
    if inData[0,tree['spInd']] > tree['spVal']:
        if type(tree['left']).__name__=='dict':
            return treeForeCast(tree['left'], inData)
        else: 
            return tree['left']
    else:
        if type(tree['right']).__name__=='dict': 
            return treeForeCast(tree['right'], inData)
        else: 
            return tree['right']
    
        
def createForeCast(tree, testData):
    m=np.shape(testData)[0]
    yHat =[]
    for i in range(m):
        yHat.append(modelTreeEval(treeForeCast(tree, testData[i,:]),testData[i,:-1]))
    return yHat


def comprRegBike(testData,testTrees):
    plotData(testData)
    Y = createForeCast(testTrees,testData)
    Y1 =testData.copy()
    print(np.corrcoef(Y,testData[:,1].T,rowvar =0)[0,1]) 
    Y1[:,1] = np.mat(Y).T
    plotData(Y1,c='r')
   
if __name__ == '__main__':
    trainingMatBike = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMatBike = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    trainingTreeBike = createTree(trainingMatBike,modelLeaf,modelErr,ops=(1,20)) 
    comprRegBike(trainingMatBike,trainingTreeBike)    
    comprRegBike(testMatBike,trainingTreeBike)   
    trainingTreeBike1 = createTree(trainingMatBike,ops=(1,20))  
    comprRegData(trainingMatBike,trainingTreeBike1)    
    comprRegData(testMatBike,trainingTreeBike1)        
    

    
    
    
    
    

