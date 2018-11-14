# -*- coding: utf-8 -*-
"""
KNN近邻算法, 分类判别
输入: inX: 测试集元素(以判定所属类)，1*N
     dataSet: 训练集样本(样本数为M, M个1*N元素) M*N
     labels:  训练集样本对应的分类 1*M
     上述均为numoy数组，否则会返回错误
     k: 近邻算法的临近个数，k<=M
输出: inX最有可能的分类个数
Created on Tue Apr 17 13:16:47 2018

@author: hexin
"""
import numpy as np
import operator
import matplotlib.pyplot as plt
from os import listdir

def classify0(inX, dataSet, labels, k):
    """ 对测试数据到训练集的按欧式距离排序，由k值得到最终的预测分类 """
    dataSetsize = dataSet.shape[0]
    ###欧式距离
    diffMat = np.tile(inX,(dataSetsize,1))-dataSet #np.tile是对数组进行广播,指定行列次数
    sqdiffMat = diffMat**2  #得到的为M*N的数组
    sqdistance = np.sqrt(np.sum(sqdiffMat,axis=1))#直接比较欧式距离的平方大小，减少计算量,返回一个一维数组
    sortedSqdistInd = sqdistance.argsort()#对一维数组进行从小到大排序，返回索引
    #利用字典搜寻排序
    classCount={}          
    for i in range(k):
        votelabel = labels[sortedSqdistInd[i]]
        classCount[votelabel] = classCount.get(votelabel,0) + 1 
        #dict.get(key, default=None) key字典中要查找的键.default如果指定键的值不存在时,返回该默认值值。
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
##利用列表排序, 比较测试字典搜寻稍快
#    classcountKey=[]; classcountValue=[]
#    for i in range(k):
#        votelabel = labels[sortedSqdistInd[i]];
#        if votelabel not in classcountKey:
#            classcountKey.append(votelabel)
#            classcountValue.append(1)
#        else:
#            votelabelInd = classcountKey.index(votelabel)
#            classcountValue[votelabelInd] += 1
#    maxCount = sorted(classcountValue,reverse=True)[0]
#    maxCountInd = classcountValue.index(maxCount)
#    return classcountKey[maxCountInd]
 
"""
利用items转化为元组迭代,比较迅速
operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号。
a = [1,2,3]  b=operator.itemgetter(1)      //定义函数b，获取对象的第1个域的值
>>> b(a) 
2
要注意，operator.itemgetter函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值。
sorted函数用来排序，sorted(iterable[, cmp[, key[, reverse]]])
其中key的参数为一个函数或者lambda函数。所以itemgetter可以用来当key的参数
a = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
根据第二个域和第三个域进行排序 sorted(students, key=operator.itemgetter(1,2))
"""


def creatDataset():
    #创造一个训练集合(测试函数分类函数的正确性)
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
#group, labels = creatDataset()
#endclass = classify0([0,0],group,labels,3)    
    
def fileDeal(filename):
    #读取文本数据进行处理,返回为数组存贮训练样本集, 和对应分类列表 
    #对应三列分别为: 每年飞行里程数；玩视频游戏所占时间；消费的冰淇淋公升数
    fread = open(filename)
    listLines = fread.readlines() #读取文本存为列表，每一行为一个列表元素
    numberOfLines = len(listLines)
    returnArray = np.zeros((numberOfLines,3)) #数据为3列
    classLabelVetore = [] #存贮对应的分类
    index = 0
    for line in listLines:
        line = line.strip() #消除空格
        listFromLine = line.split('\t') #以制表符分离数据为列表
        returnArray[index,:] = listFromLine[0:3]
        classLabelVetore.append(listFromLine[-1])
        index += 1
    return returnArray, classLabelVetore
    
dateArray, dateLabel = fileDeal('datingTestSet2.txt')
    
def dataPlot(dateArray,dateLabel):
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221)
    ax1.scatter(dateArray[:,1],dateArray[:,2])
#    ax1.axis([-2,25,-0.2,2.0])
    ax2=fig.add_subplot(222)
    ax2.scatter(dateArray[:,1],dateArray[:,2],s=40.0*np.array(dateLabel).astype(np.int),
                c= 20.0*np.array(dateLabel).astype(np.int),alpha=1) 
    #根据标签点对数据进行分类绘制
    ax3 = fig.add_subplot(223)
    ax3.scatter(dateArray[:,0],dateArray[:,1],c= 30.0*np.array(dateLabel).astype(np.int))
    ax3.axis([-5000,100000, -2,25])
    type1 = ax3.scatter([-10], [-10], s=20, c='30')
    type2 = ax3.scatter([-10], [-15], s=30, c='60')
    type3 = ax3.scatter([-10], [-20], s=50, c='90')
    ax3.legend([type1, type2, type3], ["class 1", "Class 2", "Class 3"], loc=2)
    plt.show()
    
#dataPlot(dateArray,dateLabel)
    

##由于飞行里程数值较大，其差值对最终临近点距离的影响会远远大于其余两个，而我们认为三个数据对结果影响相同
##就需要对原始数据进行归一化处理 newvalue=(oldvalue-min)/(max-min)原始数据在总数据的百分比位置
def autoNorm(dataSet):
    #归一化数据，返回处理过的数据
    minVals = dataSet.min(0) #返回每一列最小值
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataset = np.zeros_like(dataSet)
    m = dataSet.shape[0]
    normDataset = dataSet -np.tile(minVals,(m,1))
    normDataset = normDataset/(np.tile(ranges,(m,1)))
    return normDataset, ranges, minVals #返回后两个值是为了泛化时，对数据正则化求解


def dateClassTest():
    #按照一定比列对原始数据进行分类，分为训练类和测试类
    classRation = 0.1 #总数据的20%作为测试类
    dateArray, dateLabel = fileDeal('datingTestSet2.txt') #读取数据
    normDataset = autoNorm(dateArray) #归一化
    m = normDataset.shape[0]
    numTestclass = int(m*classRation) #得到测试类的个数
    countTestFalse = 0
    for i in range(numTestclass):
        TestclassResult = classify0(normDataset[i,:], normDataset[numTestclass:m,:],
                                    dateLabel[numTestclass:m] , 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (int(TestclassResult), int(dateLabel[i])))
        if TestclassResult !=  dateLabel[i]:
            countTestFalse += 1
    errorRation = countTestFalse/numTestclass
    print ('the errorRation of Testclass is: %.5f' %errorRation)
#dateClassTest()  #得到测试类的错误率，以修正参数


def classifyPerson():
    resultlist = ['not at all', 'in small doses', 'in large doses']
    gameTime = float(input("the hours you spent playing games everyday: "))
    iceCream = float(input('liters of ice_cream consumed per year: '))
    fmiles= float(input('the miles earned per year: '))
    dateArray, dateLabel = fileDeal('datingTestSet2.txt') #读取数据
    normDataset,ranges, minVals= autoNorm(dateArray) #归一化
    inArr = np.array([fmiles, gameTime, iceCream])
    inArr = (inArr-minVals)/ranges
    classifyResult = int(classify0(inArr, normDataset,dateLabel , 3))
    print('You may probably like this person: ', resultlist[classifyResult-1])
    
#classifyPerson()
def img2vector(filename): 
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handWritingTest():
    #利用k-近临点算法对数字识别训练测试
    errorCount = 0
    trainingLabel = []
    trainingFileList = listdir('digits/trainingDigits')
    m=len(trainingFileList)
    trainingVetor = np.zeros((m,1024))
    for i in range(m):
        trainingVetor[i,:] = img2vector('digits/trainingDigits/'+trainingFileList[i])
        trainingLabel.append(int(trainingFileList[i][0]))
    testFileList = listdir('digits/testDigits')
    n=len(testFileList)
    for i in range(n):
        testVetor = img2vector('digits/testDigits/'+testFileList[i])
        testLabel = int(testFileList[i][0])
        resultLabel = int(classify0(testVetor, trainingVetor, trainingLabel, 4))
        if resultLabel != testLabel:
            errorCount += 1
    errorTest = errorCount/n
    print('the error of numTest is: %.6f' %errorTest)
handWritingTest()
        
    




  













    