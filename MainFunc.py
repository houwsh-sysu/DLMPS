#!/usr/bin/env python
# coding: utf-8

# In[ ]:

######################ver2.0
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pylab

import time
from PIL import Image
import random
from pathlib2 import Path#python3环境下
#from pathlib import Path  #python2环境下
import os
from tvtk.api import tvtk, write_data
import threading
from time import sleep
from tqdm import tqdm
import difflib
import itertools as it
#################################本程序自带子程序########################
from zuobiao import*
from Pdatabase import*
from patterntool import*
from editdistance import*
from roadlist import*
import heapq
from listheap import*
from initialGrid import*
from Patchmatch import*
from cluster import*
from AIinitial import*
from Fault import*
from Fault2 import*
from NewEM import*
from fenji import*
from Pythia import *

from Initial import *

################################计时程序########################################################
time_start1=time.time()#计时开始


######################################参数读取阶段################################################################



file1=open('./parameter.txt')
content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
Mh=int(''.join(string1))
#print Mh

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
Mx=int(''.join(string1))
#print Mx

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
My=int(''.join(string1))
#print My

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
lag=int(''.join(string1))
#print lag

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
lag_h=int(''.join(string1))
#print lag_h

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
lag_x=int(''.join(string1))
#print lag_x

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
lag_y=int(''.join(string1))
#print lag_y

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
patternSizeh=int(''.join(string1))
#print patternSizeh

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
patternSizex=int(''.join(string1))
#print patternSizex

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
patternSizey=int(''.join(string1))
#print patternSizey

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
U=int(''.join(string1))
#print U

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
N=int(''.join(string1))
#print N

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
size=int(''.join(string1))
#print size

content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
itr=int(''.join(string1))
#print itr


content=file1.readline()
scale=[]
for i in content:
    if str.isdigit(i):
        scale.append(int(i))
#print scale

#一次模拟的个数
content=file1.readline()
string1=[i for i in content if str.isdigit(i)]
Modelcount=int(''.join(string1))
#print Modelcount
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
#####################预处理阶段################################################################
epoch=10000

valueliata=[]
valueliata=Tilistvalueextract()
valueliata.sort(reverse=True)
flaglist=[]#用来判断地层为局部或者全局
print (valueliata)
valueliata=[]
print('请自行进行排序:',valueliata)
code=[]
hardlist=[]
jvalue=29
m=-np.ones((Mh,Mx,My),int)#默认空值为-1

Pythia(m,patternSizeh,patternSizex,patternSizey,lag,lag_h,lag_x,lag_y,N,U,hardlist,code,valueliata,scale,epoch,flaglist,jvalue)

time_end1=time.time()#计时结束
print('总耗时：',time_end1-time_start1)


################################################
def Pythia(m, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, N, U, hardlist, code, valueliata, scale,
           epoch, flaglist, jvalue):
    # 皮提亚主程序
    # code 需要事先定义好TI中不包含的

    # m,Tilist,Tizuobiaolist=sectionloadandextend(m,patternSizex,patternSizey,0,1)
    m, Tilist, Tizuobiaolist, codelist = sectionloadandextendG(m, template_x, template_y, flag, 1, jvalue)
    # 待插入二维高程建模区域约束
    codelist.append(code)

    print('当前合理地层层序列表：', codelist)
    print('当前模拟地层顺序：', valueliata)
    m = initialAIforPythia(m, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, N, U, hardlist, codelist,
                           valueliata)
    # 初始化部分结束

    # em迭代部分

    sancheck = 1  # sectionloadandextend倍率机制
    np.save('./output/initial.npy', m)
    # EM迭代阶段
    for ni in range(len(scale)):
        sancheck = sancheck * scale[ni]
        # 构建新初始网格mm
        mm = -np.ones((int(m.shape[0] * scale[ni]), int(m.shape[1] * scale[ni]), int(m.shape[2] * scale[ni])), int)

        Tilist = []
        Tizuobiaolist = []
        mm, Tilist, Tizuobiaolist, codelist = sectionloadandextendG(mm, patternSizex, patternSizey, 1, sancheck, jvalue)

        mm = extendTimodel(mm, patternSizeh, patternSizex, patternSizey)
        # 上一个尺度升采样
        m = simgridex(m, scale[ni])
        # 重新导入
        for hi in range(m.shape[0]):
            for xi in range(m.shape[1]):
                for yi in range(m.shape[2]):
                    if mm[hi, xi, yi] != -1:
                        m[hi, xi, yi] = mm[hi, xi, yi]
        print("该尺度扩展Ti完成")
        time_start = time.time()  # 计时开始
        # np.save('./output/m1.npy',m)
        #########m= patchmatchmultiTiB(m,Tilist,size,itr,1)
        CTI = []  # 检测使用率剖面
        # m,CTI= patchmatchmultiTiBZ2ver(m,mm,Tilist,size,itr,1)
        m, CTI = Recodepatchmatch(m, mm, Tilist, Tizuobiaolist, size, itr, 4, 0)  # 并行进程的数目
        # 计算量加大过多
        # m,CTilist= patchmatchmultiTiBZzuobiaover(m,mm,Tilist,Tizuobiaolist,size,itr,1)
        path = "./output/reconstruction.npy"
        np.save(path, m)
        time_end = time.time()
        # size=size*scale[ni]+1
        print("该尺度优化完成")
        print('timecost:')
        print(time_end - time_start)

        data = m.transpose(-1, -2, 0)  # 转置坐标系
        grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                              dimensions=data.shape)
        grid.point_data.scalars = np.ravel(data, order='F')
        grid.point_data.scalars.name = 'lithology'
        write_data(grid, './output/output.vtk')
    return m