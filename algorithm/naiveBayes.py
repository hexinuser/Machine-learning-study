# -*- coding: utf-8 -*-
"""
朴素贝叶斯
P(C|X)=P(X|C)*P(C)/P(X)
C为类别，X为样品
Created on Fri Apr 20 15:09:34 

@author: hexin
"""

import numpy as np
import math

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 表示侮辱, 0 为非侮辱
    return postingList,classVec
                 
def createVocabList(dataSet):
    #将所有数据单词唯一存储在一个列表中
    vocabSet = set([])  #必须设置为set与下列
    for document in dataSet:
        vocabSet = vocabSet | set(document) #set集的交并会变成set集合
    return list(vocabSet)
#def createVocabList0(dataSet):
#    #两种不同的存贮转换方式，但此种方式最后对列表去重，相对时间较长
#    List0=[]  
#    for List in dataSet:
#        List0.extend(set(List))
#    return set(List0)

def setOfWords2Vec(vocabList, inputSet):
    #判断输入语句的单词是否在存储的列表中，生成一个与存储列表大小相同的列表，对应位置值为0或1，表示是否存在
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainDataToMat(dataSet):
    trainMat = []
    dataList = createVocabList(dataSet)
    for dataset in dataSet:
        trainMat.append(setOfWords2Vec(dataList,dataset))
    return trainMat,dataList
        

def trainProbWord(trainMat,trainCateg):
    #trainMat为一个列表，列表的每个元素就是一个数值列表表示一个句子，trainCateg表示该语句是否为侮辱性列表
    #得到贝叶斯中P(X|C), 即在不同类别中，每个单词的概率组成的行向量
    numOfTrain = len (trainMat) #训练样本个数
    pFaultWord = sum(trainCateg)/numOfTrain  #所有语句中侮辱想词语的比列，表示P(C)
    numOfWord = len(trainMat[0]) #样本中所有的单词数，来判断每个单词是否为侮辱性的概率
#    p1NumWord = np.zeros(numOfWord)
#    p0NumWord = np.zeros(numOfWord)   
#    p0SumWord = 0.0; p1SumWord = 0.0
    p1NumWord = np.ones(numOfWord)
    p0NumWord = np.ones(numOfWord)  #初始化数值为1，防止某个概率值为0，乘积为0 
    p0SumWord = 2.0; p1SumWord = 2.0#初始化数值为2，对应初始值数值为1trainMat
    for i in range(numOfTrain):
        if trainCateg[i] == 1:
            p1NumWord += trainMat[i]
            p1SumWord += sum(trainMat[i])
        else:
            p0NumWord += trainMat[i]
            p0SumWord += sum(trainMat[i])
    p1VecForWord = np.log(p1NumWord/p1SumWord)
    p0VecForWord = np.log(p0NumWord/p0SumWord)
    return pFaultWord, p1VecForWord,p0VecForWord
#trainSet,classVec = loadDataSet()
#trainMat,dataList = trainDataToMat(trainSet)
#pFaultWord, p1VecForWord,p0VecForWord =trainProbWord(trainMat,classVec) 

def classify0(classifyVec,pFaultWord, p1VecForWord,p0VecForWord) :
    p1 = np.dot(classifyVec,p1VecForWord) + math.log(pFaultWord) #对数化计算贝叶斯的乘积，一因为分母相同，可不计算
    p0 = np.dot(classifyVec,p0VecForWord) + math.log(1-pFaultWord)
    if p1 > p0:
        return 1
    else:
        return 0       
    
#####上述为训练集产生产生结果的函数
        
def repeatWordVec(dataList,wordList):
    wordArray = np.zeros(len(dataList))
    for word in wordList:
        if word in dataList:
            wordIndex = dataList.index(word)
            wordArray[wordIndex] += 1   #多一个单词多次出现，记次数而不是单纯定义为1
    return wordArray

#trainSet,classVec = loadDataSet()
def resultClassify(wordList, trainSet, classVec):
    trainMat,dataList = trainDataToMat(trainSet)
    pFaultWord, p1VecForWord,p0VecForWord =trainProbWord(trainMat,classVec) 
    wordArray = repeatWordVec(dataList,wordList)
        
    return classify0(wordArray,pFaultWord, p1VecForWord,p0VecForWord)
        
        
def textParse(bigString):
    #利用re模块将从文本中读取的数据，仅仅读取其单词并存储为列表
    import re
    #以非字母为下划线的进行划分
    listOfString = re.split(r'\W',bigString)
    #去除空字符串
    return [word.lower() for word in listOfString if len(word)>2] 

def sepSet():
    from os import listdir
    dataSet = [];  spamOrNot = []
    spamDir = listdir('email/spam')
    spamNotDir = listdir('email/ham')
    for spamName in spamDir:
        fr = open('email/spam/'+spamName)
        sapmList = textParse(fr.read())
        dataSet.append(sapmList)
        spamOrNot.append(1)
    for spamNotName in spamNotDir:
        fr = open('email/ham/'+spamNotName)
        spamNotList = textParse(fr.read())
        dataSet.append(spamNotList)
        spamOrNot.append(0)
    n = len(dataSet)
    testIndex = np.random.permutation(range(n))[:10]##随机获得10个样本作为测试集，#permutation打乱列表顺序
    testData = []; testSpamOrNot = []
    trainData = []; trainSpamOrNot = []
    for i in range(n):
        if i in testIndex:
            testData.append(dataSet[i])
            testSpamOrNot.append(spamOrNot[i])
        else:
            trainData.append(dataSet[i])
            trainSpamOrNot.append(spamOrNot[i])
    testClassResult = []
    count = 0
    for i in range(len(testData)):
        testresult = resultClassify(testData[i], trainData, trainSpamOrNot)
        if testresult != testSpamOrNot[i]:
            count +=1
        testClassResult.append(testresult)
    error = count/len(testClassResult)
    return error,testClassResult,testSpamOrNot
error,testClassResult,testSpamOrNot =sepSet()
#error = 0
#for i in range(100):
#    error += sepSet()[0]
#error = error/100


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
    
 