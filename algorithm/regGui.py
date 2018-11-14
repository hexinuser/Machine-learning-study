# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:13:52 2018

@author: hexin
"""

import tkinter as tk 
import numpy as np
import treeReg

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS,tolN):
    reDraw.f.clf()        # clear the figure
    reDraw.ax = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: tolN = 2
        myTree=treeReg.createTree(reDraw.rawDat, treeReg.modelLeaf,\
                                   treeReg.modelErr, (tolS,tolN))
        yHat = treeReg.treeForeCast(myTree, reDraw.testDat )
    else:
        myTree=treeReg.createTree(reDraw.rawDat, ops=(tolS,tolN))
        yHat = treeReg.treeForeCast(myTree, reDraw.testDat)
    reDraw.ax.scatter(reDraw.rawDat[:,0], reDraw.rawDat[:,1], s=5) #use scatter for data set
    reDraw.ax.plot(reDraw.testDat, yHat, linewidth=2.0) #use plot for yHat
    reDraw.canvas.show()

def getInputs():
    try: tolN = int(tolNentry.get())
    except: 
        tolN = 10 
        print ("enter Integer for tolN")
        tolNentry.delete(0, tk.END)
        tolNentry.insert(0,'10')
    try: tolS = float(tolSentry.get())
    except:  #无输入，或输入格式错误
        tolS = 1.0 
        print ("enter Float for tolS")
        tolSentry.delete(0, tk.END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS

def drawNewTree():
    tolN,tolS = getInputs()#get values from Entry boxes
    reDraw(tolS,tolN)





root = tk.Tk()
#myLabel = tk.Label(root,text = 'hello world')#标签的创建
#myLabel.grid() #标签存在一个二维表格，并可以指定显示位置
#root.mainloop()  #建立窗口，并显示出结果


reDraw.f = Figure(figsize=(5,4), dpi=100) #建立全局的图形窗口
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
tk.Label(root,text = 'Plot Place Holder').grid(row = 0, columnspan = 3)
tk.Label(root, text="tolN").grid(row=1, column=0)
tk.Label(root, text="tolS").grid(row=2, column=0)


tolNentry = tk.Entry(root)  #文本输入框
tolNentry.grid(row=1, column=1)
tolNentry.insert(0,'10')  #文本 框默认输入值

tolSentry = tk.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0,'1.0')

tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = tk.IntVar()
chkBtn = tk.Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = np.mat(treeReg.loadDataSet('sine.txt'))
reDraw.testDat = np.arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0, 10)
#               
root.mainloop()