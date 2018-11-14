# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:24:11 2018

@author: hexin
"""

import matplotlib.pyplot as plt
import dTrees
from pylab import mpl 
mpl.rcParams['font.sans-serif'] = ['SimHei']  
#解决绘图中文乱码的问题

decisionNode = dict(boxstyle="sawtooth", fc="0.8")#节点绘图属性
leafNode = dict(boxstyle="round4", fc="0.8")#叶节点绘图属性
arrow_args = dict(arrowstyle="<-") #连接线的属性，箭头属性设置


def getNumLeaf(mytree):
    #叶节点个数
    numOfLeaf = 0
    fristLeaf = list(mytree.keys())[0]
    secondTree = mytree[fristLeaf] #因tree的存贮，是属性和属性值对应字典，所以每个secondTree都是一个字典
    for key in secondTree.keys():
        if type(secondTree[key]).__name__ == 'dict':#判断属性对应属性值是否含有字典嵌套，以此判断是否为叶节点
            numOfLeaf += getNumLeaf(secondTree[key])
        else:
            numOfLeaf += 1
    return numOfLeaf


def getTreeDepth(mytree):
    #决策树深度
    maxDepth = 0
    fristKey = list(mytree.keys())[0]
    secondTree = mytree[fristKey]
    for key in secondTree.keys():
        if type(secondTree[key]).__name__ == 'dict':
            branchDepth = 1+ getTreeDepth(secondTree[key])
        else:
            branchDepth = 1
        if branchDepth > maxDepth:
            maxDepth = branchDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    #绘制节点，分别对应节点文本，文本中心点，初始点，右初始点箭头指向中心点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    ##xycoords的axes fraction表示坐标点（0，0）在左下角，（1，1）在右上角
#def createPlot0():
#    fig = plt.figure(1, facecolor='white')
#    #由ceratPlot.ax1建立一个全局绘图区
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)#最后一个参数定义绘图框类型
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()


def plotMidText(cntrPt, parentPt, txtString):
    #在父子节点坐标连线中间位置添加对应的属性值字符串，对应子节点坐标，父节点坐标，和对应需添加的文本值
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0] #在父子节点中间输入属性值
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1] #xmid和ymid是绘制点坐标
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeaf(myTree)  #得到每一次迭代根节点的叶节点的个数来决定行坐标
    firstStr =list(myTree.keys())[0]   #父结点文本
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)#当前结点的坐标
    #每出现一个叶节点，下一个结点的位置移动t/n(前方出现的叶节点个数再加上，1/2n(n为总的叶节点个数)
    plotMidText(cntrPt, parentPt, nodeTxt)  #对父节点上方，无属性值，输入为空
    plotNode(firstStr, cntrPt, parentPt, decisionNode)#绘制节点，对父节点来说起始点和终止点相同，绘制的箭头没有
    secondDict = myTree[firstStr] #仍为一个字典，键值为对应的属性值
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #根据树的深度，调节y的坐标
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':# 判断对应属性是否为叶结点
            plotTree(secondDict[key],cntrPt,str(key))        #不为叶节点
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW #每个叶节点间隔1/n(n为叶节点个数)
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))  #绘制结点的属性值
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #无坐标轴
#    createPlot.ax1 = plt.subplot(111, frameon=False) #有坐标轴 
    plotTree.totalW = float(getNumLeaf(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')#从坐标（0.5,1.0)开始绘制
    plt.show()

    
#dataSet, labels = dTrees.createDataSet()
#mytree =  dTrees.creatTrees(dataSet,labels)  
##mytree['no surfacing'][3] = 'maybe' 
#createPlot(mytree)



