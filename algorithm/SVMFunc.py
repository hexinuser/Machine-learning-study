# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:31:10 2018

@author: hexin
"""

import numpy as np
import matplotlib.pyplot as plt
#import math
#全部设置成矩阵的形式方便乘积

def loadTestSet(filename):
    #读取数据
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        lineArr = list(map(lambda x: float(x),lineArr)) #对数字字符串转化为浮点数存储
        dataMat.append(lineArr[:-1])
        labelMat.append(lineArr[-1])
    return dataMat, labelMat
        

def selectJrand(i,m):
#    随机选取一个和i不同的j值
    j = i
    while j==i:
        j = int(np.random.uniform(0,m))
    return j

def adjVar(alpha,L,H):
    #将alpha限制投影到[L,H]上
    if alpha<L:
        return L
    if alpha>H:
        return H
    return alpha


def calcW(alpha,dataMat,classLabel):
    #根据所得数据计算w,返回的是一个一维数组
    dataMat = np.mat(dataMat); classLabel = np.mat(classLabel).T  
    w = dataMat.T* np.multiply(alpha,classLabel)
    return w.A.flatten()

def plotSVM(alpha,b,dataMat,labelMat,w):
    ##对二维点线性核就行绘制点及分隔
    supVec = np.nonzero(alpha.A>0)[0].tolist()#支撑向量的点索引
    supDataX = list(map(lambda i: dataMat[i][0],supVec))
    supDataY = list(map(lambda i: dataMat[i][1],supVec))
    dataMat = np.mat(dataMat) ; 
    n = len(labelMat)
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
    ax.scatter(supDataX,supDataY,s=240,marker='o',c='',edgecolors='black')
    ##标记支持向量的点,空心圆
    x = np.arange(-5,10.0,0.1)
    y = (-b-w[0]*x)/w[1]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.xlim(-3,12)
    plt.ylim(-10, 8)    #设置坐标显示范围
    plt.show()

"""
######################
######################
######################
"""


#def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
#    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).T
#    b = 0; m,n = np.shape(dataMatrix)
#    alphas = np.mat(np.zeros((m,1)))
#    iter = 0
#    #当更新次数达到最大次数是，停止计算，计算量较大
#    while (iter < maxIter):
#        alphaPairsChanged = 0
#        for i in range(m):
#            fXi = np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T) + b
#            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
#            ###判断i是否违背KKT条件
#            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
#                j = selectJrand(i,m) #随机选取j值进行更新
#                fXj = np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T) + b
#                Ej = fXj - float(labelMat[j])##计算Ej误差
#                
#                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
#                if (labelMat[i] != labelMat[j]):
#                    L = max(0, alphas[j] - alphas[i])
#                    H = min(C, C + alphas[j] - alphas[i])
#                else:
#                    L = max(0, alphas[j] + alphas[i] - C)
#                    H = min(C, alphas[j] + alphas[i])
#                if L==H: 
##                    print ("L==H"); 
#                    continue
#                #选取的j值不能进行更新，重新选取第一个变量i
#                
#                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
#                if eta == 0: 
##                    print ("eta=0"); 
#                    continue
#                
#                #更新alpha[i]和alpha[j]
#                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
#                alphas[j] = adjVar(alphas[j],L,H)
#                if (abs(alphas[j] - alphaJold) < 0.00001):
##                    print ("j not moving enough"); 
#                    continue
#                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
#                                                                        #the update is in the oppostie direction
#                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
#                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
#                if (0 < alphas[i]) and (C > alphas[i]): 
#                    b = b1
#                elif (0 < alphas[j]) and (C > alphas[j]): 
#                    b = b2
#                else: 
#                    b = (b1 + b2)/2.0
#                alphaPairsChanged += 1
##                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
#        if (alphaPairsChanged == 0): 
#            iter += 1 #当对每个i都不再更新时，增加一次迭代次数（j的随机性选取可能导致违背KKT的i也不更新）
#        else:
#            iter = 0
##        print ("iteration number: %d" % iter)
#    return float(b),alphas
#
#
#
#filename = 'testSet.txt'
#dataMat, labelMat = loadTestSet(filename)
#b,alpha = smoSimple(dataMat, labelMat, 3, 0.001, 40)
#w =  calcW(alpha,dataMat,labelMat)
#print(w); print(b)
#plotSVM(w,b,dataMat)

"""##############
上述是一个简化的SOM方法解决SVM，其缺点是尽量多的更新使得达到
最优解，当往往对某些数据并不需要迭代那么多次，导致迭代次数多，而
每一步迭代的计算量比较大，就导致最终计算的时间较长
################"""


def kernelFun(fname,X,Y):
    ##fname为核函数名称及其对应所需参数，X，Y为行矩阵
    if fname[0] == 'lin': ##线性核
        return X*Y.T
    elif fname[0] == 'poly': ##多项式核,fanme[1]=d>=1为多项式次数
        return (X*Y.T)**fname[1]
    elif fname[0] == 'gauss':  ##高斯核,fname[1]=sigma>0带宽
        sigma = fname[1]
        return np.exp(-(X-Y)*(X-Y).T/(2*sigma**2))
    elif fname[0] == 'laplace':  ##拉普拉斯核，fname[1]=sigma>0
        return np.exp(-np.sqrt(((X-Y)*(X-Y).T))/fname[1])
    elif fname[0] == 'sigmoid': ##sigmoid核函数,fname[1]和[2]对应两个参数beta和theta
        return np.tanh(fname[1]*X*Y.T+fname[2])
        
        
def kernelMethon(Opt,fname):
    #dataMat为原数据集，kernerFun为对应的核函数
    ##利用核函数方法计算出数据集的核矩阵，计算时，直接调用
    for i in range(Opt.m):
        for j in range(i,Opt.m):
           Opt.K[i,j] = kernelFun(fname,Opt.dataMat[i,:],Opt.dataMat[j,:])
           Opt.K[j,i] = Opt.K[i,j]
#    print(Opt.K)

class optPara:
    #参数类的设置，方便修改
    def __init__(self,dataMat,classLabel,C,tol):
        self.dataMat = dataMat
        self.classLabel = classLabel#classLabel为一个列矩阵
        self.C = C
        self.tol = tol
        self.m = np.shape(dataMat)[0] #数据量的个数
        self.alpha = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.activeYN = np.mat(np.zeros((self.m,2))) ##记alpha是否更改，
        #更改的可能在间隔边界上，寻找第二个变量可减少计算
        self.K =  np.mat(np.zeros((self.m,self.m)))
        

        
        
        
def calcEi(Opt,i):
    ##Opt表示设置的参数类，返回Ei,得到的是第i个数据集的预测值和真实值的误差
#    fxI = np.multiply(Opt.alpha,Opt.classLabel).T*(Opt.dataMat*Opt.dataMat[i,:].T)+Opt.b
    fxI = np.multiply(Opt.alpha,Opt.classLabel).T*Opt.K[:,i]+Opt.b
    Ei = fxI - Opt.classLabel[i]
    return Ei
 


def chooseSecVar(oS,i, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.activeYN[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.activeYN[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEi(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j= selectJrand(i,oS.m)
    return j, calcEi(oS,j)
##        


def updataYN(Opt,i):
    #对更新的i进行标记
    Ei =calcEi(Opt,i)
    Opt.activeYN[i] = [1,Ei]


def updateAlpha(Opt,i):
    #根据i,判断是否违背KKT条件(含一定的容许误差)，可否选择为第一个变量，再选取j值，然后进行更新
    ## 返回值0,1表示是否更新了对应的alpha值
    Ei = calcEi(Opt,i)
    if (Opt.alpha[i]<Opt.C and Opt.classLabel[i]*Ei<-Opt.tol) or\
    (Opt.alpha[i]>0 and Opt.classLabel[i]*Ei>Opt.tol): #违背了KKT条件
        j,Ej = chooseSecVar(Opt,i,Ei)
        if Opt.classLabel[i] == Opt.classLabel[j]:
            L = max(0,Opt.alpha[j]+Opt.alpha[i]-Opt.C)
            H = min(Opt.C,Opt.alpha[i]+Opt.alpha[j])
        else:
            L = max(0,Opt.alpha[j]-Opt.alpha[i])
            H = min(Opt.C,Opt.C+Opt.alpha[j]-Opt.alpha[j])
        if L==H:
            return 0 #相等时不进行更新，退出函数
        #得到两个线性核向量的差值，其他核函数对应更改
#        normIJ = (Opt.dataMat[i,:]-Opt.dataMat[j,:])*(Opt.dataMat[i,:]-Opt.dataMat[j,:]).T
        normIJ = Opt.K[i,i]+Opt.K[j,j]-2*Opt.K[i,j]
        if normIJ==0:
            #此时不存在和i不同的数据，不更新
            return 0
        alphaJold =  Opt.alpha[j].copy(); alphaIold =  Opt.alpha[i].copy()
        Opt.alpha[j] += Opt.classLabel[j]*(Ei-Ej)/normIJ
        Opt.alpha[j] = adjVar(Opt.alpha[j],L,H) #将更新的值剪辑约束到集合内
        updataYN(Opt,j)
        if (abs(Opt.alpha[j]-alphaJold)) < 1e-5:
#            print('j not moving enough')
            return 0 #alpha[j]更新不够，不进行其他变量的更新
        Opt.alpha[i] += Opt.classLabel[j]* Opt.classLabel[i]*(alphaJold-Opt.alpha[j])
        updataYN(Opt,i)
        
#        biNew = -Ei-Opt.classLabel[i]*(Opt.dataMat[i,:]*Opt.dataMat[i,:].T)*(Opt.alpha[i]-alphaIold)-\
#        Opt.classLabel[j]*(Opt.dataMat[i,:]*Opt.dataMat[j,:].T)*(Opt.alpha[j]-alphaJold)+Opt.b
#        
#        bjNew = -Ej-Opt.classLabel[i]*(Opt.dataMat[i,:]*Opt.dataMat[j,:].T)*(Opt.alpha[i]-alphaIold)-\
#        Opt.classLabel[j]*(Opt.dataMat[j,:]*Opt.dataMat[j,:].T)*(Opt.alpha[j]-alphaJold)+Opt.b
        
        biNew = -Ei-Opt.classLabel[i]*(Opt.K[i,i])*(Opt.alpha[i]-alphaIold)-\
        Opt.classLabel[j]*(Opt.K[i,j])*(Opt.alpha[j]-alphaJold)+Opt.b
        
        bjNew = -Ej-Opt.classLabel[i]*(Opt.K[i,j])*(Opt.alpha[i]-alphaIold)-\
        Opt.classLabel[j]*(Opt.K[j,j])*(Opt.alpha[j]-alphaJold)+Opt.b
        
        if Opt.alpha[j]>0 and Opt.alpha[j]<Opt.C:
            Opt.b = bjNew
        elif Opt.alpha[i]>0 and Opt.alpha[i]<Opt.C:
            Opt.b = biNew
        else:
            Opt.b = (biNew+bjNew)/2
        return 1
    else:
        return 0

def smoFunc(dataMat, labelMat, C, tol,maxIter = 50,kernelName = ['lin']):
    Opt = optPara(np.mat(dataMat),np.mat(labelMat).T,C,tol)
    kernelMethon(Opt,kernelName)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            #遍历所有样本，是否满足KKT条件，选取j值更新
            for i in range(Opt.m):        
                alphaPairsChanged += updateAlpha(Opt,i)
#                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:
            #当遍历结束后，在对间隔边界的支持向量点（0<alpha<C）遍历验证，若都满足，再重新
            #遍历整个样本，直至都满足
            nonBoundIs = np.nonzero((Opt.alpha.A > 0) * (Opt.alpha.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += updateAlpha(Opt,i)
#                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: 
            #遍历结束，验证间隔边界的点
            entireSet = False #i遍历一轮，初始时随机选择j
        elif (alphaPairsChanged == 0): 
            entireSet = True  
            #间隔边界的点都满足KKT，重新验证数据集
#        print ("iteration number: %d" % iter)
    return Opt.b,Opt.alpha
                  


#filename = 'testSet.txt' 
#dataMat, labelMat = loadTestSet(filename)
#b,alpha = smoFunc(dataMat, labelMat, 0.6, 0.001)
#w =  calcW(alpha,dataMat,labelMat)
##print(w); print(b);print(iter)
#plotSVM(alpha,float(b),dataMat,labelMat,w)
#    
def calcFun(testVec,supMat,supLabel,supAlpha,b,kernelName):
    m = np.shape(supAlpha)[0]
    fx = 0.0
    for i in range(m):
        fx += float(supAlpha[i]*supLabel[i]*kernelFun(kernelName,testVec,supMat[i]))
    fx += float(b) 
    return np.sign(fx)
    
#    
#
def plotData(dataMat,labelMat):    
    #绘制数据样本点
    dataMat = np.mat(dataMat) ; 
    n = len(labelMat)
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
#    ax.scatter(supDataX,supDataY,s=240,marker='o',c='',edgecolors='black')
#    ##标记支持向量的点,空心圆

    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()




def testKernel():
    filename = 'testSetRBF.txt'
    dataMat,labelMat = loadTestSet(filename)
#    plotData(dataMat,labelMat)
    kernelName = ['gauss',1.1]
    b,alpha = smoFunc(dataMat, labelMat, 0.6, 0.001,kernelName )
    supVec = np.nonzero(alpha.A>0)[0]#支撑向量的点索引
    print('the number of support vector:%d'%(len(supVec)))
    dataMat = np.mat(dataMat); labelMat = np.mat(labelMat).T
    m,n = np.shape(dataMat)

    supAlpha = alpha[supVec]
    supMat = dataMat[supVec]; supLabel = labelMat[supVec];
    errorCount = 0 
    for i in range(m):
        fx = calcFun(dataMat[i,:],supMat,supLabel,supAlpha,b,kernelName)
        if fx != float(np.sign(labelMat[i])):
            errorCount += 1
    print ('the error rate of the test vector:%.5f'%(errorCount/m))
    filename2 = 'testSetRBF2.txt'
    dataMat2,labelMat2 = loadTestSet(filename2)
#    plotData(dataMat2,labelMat2)
    dataMat2 = np.mat(dataMat2); labelMat2 = np.mat(labelMat2).T
    m2 = np.shape(dataMat2)[0]
    errorCount2 = 0 
    for i in range(m2):
        fx = calcFun(dataMat2[i,:],supMat,supLabel,supAlpha,b,kernelName)
        if fx != float(np.sign(labelMat2[i])):
            errorCount2 += 1
    print ('the error rate of the test vector:%.5f'%(errorCount2/m2))
#testKernel()
    
def loadNumData(dirname):
    from os import listdir
    labelMat = []
    fileList = listdir(dirname)
    dataMat = []
    for filename in fileList:
        numMat = []
        fr = open(dirname+'/'+filename)
        for line in fr.readlines():
            IntList = list(map(lambda x:int(x),line[:-1]))
            numMat.extend(IntList)
        dataMat.append(numMat)
        if (int(filename[0]) == 1):
            labelMat.append(-1)
        else:
            labelMat.append(1)
    return dataMat,labelMat

def testNum():
    testDirName = 'digits/trainingDigits'
    dataMat,labelMat = loadNumData(testDirName)
    kernelName = ['gauss',100]
    b,alpha = smoFunc(dataMat, labelMat, 200, 0.0001,10000,kernelName)
    supVec = np.nonzero(alpha.A>0)[0]#支撑向量的点索引
    print('the number of support vector:%d'%(len(supVec)))
    dataMat = np.mat(dataMat); labelMat = np.mat(labelMat).T
    m,n = np.shape(dataMat)

    supAlpha = alpha[supVec]
    supMat = dataMat[supVec]; supLabel = labelMat[supVec];
    errorCount = 0 
    for i in range(m):
        fx = calcFun(dataMat[i,:],supMat,supLabel,supAlpha,b,kernelName)
        if fx != float(np.sign(labelMat[i])):
            errorCount += 1
    print ('the error rate of the test vector:%.5f'%(errorCount/m))
    testDirName2 = 'digits/testDigits'
    dataMat2,labelMat2 = loadNumData(testDirName2)
    dataMat2 = np.mat(dataMat2); labelMat2 = np.mat(labelMat2).T
    m2 = np.shape(dataMat2)[0]
    errorCount2 = 0 
    for i in range(m2):
        fx = calcFun(dataMat2[i,:],supMat,supLabel,supAlpha,b,kernelName)
        if fx != float(np.sign(labelMat2[i])):
            errorCount2 += 1
    print ('the error rate of the test vector:%.5f'%(errorCount2/m2))

testNum()

#            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    