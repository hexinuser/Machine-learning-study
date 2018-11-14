 # -*- coding: utf-8 -*-
"""
Created on Tue May  1 10:35:21 2018

@author: hexin
"""
import numpy as np
import matplotlib.pyplot as plt


def loadTestSet():
    #返回数据集数组，标签列表,数据集最后一列增加了一列单位向量，对应的w最后的一个元素为b
    dataSet = []; dataLabel = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineData = line.strip().split('\t')
        lineData = list(map(lambda x: float(x),lineData))
        dataSet.append(lineData[:-1]+[1])
        dataLabel.append(lineData[-1])
    return np.array(dataSet), dataLabel
a,b=loadTestSet()



"""
逻辑回归，通过极大化似然函数
等价于极小化
l(beta)=sum_{i=1}^m (-y_i*beta^T*x_i+ln(1+e^(beta^T*x_i)))
beta表示回归系数w和b,x_i表示（x;1）

"""


def sigmoid(inX):
    #输入numpy数组
    return (1-1.0/(1+np.exp(inX)))

def gradAscent(dataNpIn, classLabels):
    #利用梯度上升，极大化似然函数
    dataNpIn = np.mat(dataIn); classLabels = np.mat(classLabels).T
    weightsStat = []
    m,n=dataNpIn.shape
    alpha = 0.01  #步长可以采取精确或非精确搜索来寻找，而不是去固定值
    maxIter =4000
    weights = np.mat(np.ones((n,1)))
    for k in range(maxIter):    
        h = sigmoid(dataNpIn*weights)  #得到一个列矩阵，每一列元素为w*x
#        error = (classLabels - h) 
#        weights = weights + alpha * np.dot(dataNpIn.T, error)
      
        error = (classLabels - h) 
        weights = weights +alpha * dataNpIn.T*error
        weightsStat.append(weights)#将weights转化为一维数组存储
    weightsStat =np.array(weightsStat) #转化为数组存储
    return weights, weightsStat
#dataIn, dataLabels = loadTestSet()
#weights = gradAscent(dataIn, dataLabels)
def randGradAs(dataNpIn, classLabels):
    #利用随机梯度上升，极大化似然函数
    dataNpIn = np.mat(dataIn); classLabels = np.mat(classLabels).T
    m,n=dataNpIn.shape
    alpha = 0.01
    maxIter = 50000
    weights = np.mat(np.ones((n,1)))
    weightsStat =[]
    for k in range(maxIter):    
        i = np.random.randint(0,m)
        h = sigmoid(np.mat(dataNpIn[i,:])*weights)
        error = (classLabels[i] - h) 
        weights = weights +alpha *np.mat(dataNpIn[i,:]).T*error
        weightsStat.append(weights)
    
    weightsStat =np.array(weightsStat)
    return weights,weightsStat


def plotBestFit(weights):
    dataMat, labelMat = loadTestSet()
    n = len(labelMat)
    weights = list(map(lambda x: float(x),weights))
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = [] #分别存储两种不同分类的数据
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataMat[i,0])
            ycord1.append(dataMat[i,1])
        else:
            xcord2.append(dataMat[i,0])
            ycord2.append(dataMat[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='r',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='y')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[2]-weights[0]*x)/weights[1]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
            
    
def plotWIter(weightsStat):
    weights1 = weightsStat[:,0].tolist()
    weights2 = weightsStat[:,1].tolist()
    weights3 = weightsStat[:,2].tolist()
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    x = list(range(len(weights1)))
    ax1.plot(x,weights1)
    ax2.plot(x,weights2)
    ax3.plot(x,weights3)
    plt.show()
    
dataIn, dataLabels = loadTestSet()
weights ,stats= randGradAs(dataIn, dataLabels)
#weights = weights.getA()

plotBestFit(weights)
plotWIter(stats)

























