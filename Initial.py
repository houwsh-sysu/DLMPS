#!/usr/bin/env python
# coding: utf-8

# In[ ]:

####   初始化
import cv2
from Semantic import *


def sectionloadandextendG(m,template_x,template_y,flag,scale,jvalue):#flag==1为patchmatch步骤，0为initial步骤 比起上面的版本增加了剖面上方归为零值的默认操作、防止小个体jvalue被缩放忽略的操作
    #对剖面进行导入和Ti提取的函数  #scale为当前倍率
    #输出的坐标列表为RecodePatchmatch需要的格式
    #插入地层层序判断的版本 加入了h方向的坐标
    Tilist=[]
    Tizuobiaolist=[]
    codelist=[]#地层层序列表
    file1=open('./Ti/Tiparameter.txt')
    content=file1.readline()
    string1=[i for i in content if str.isdigit(i)]
    num=int(''.join(string1))
    print('剖面数目：')
    print(num)
    for n in range(num):
        guding=[]
        for aa in range(6):
            content=file1.readline()
            string1=[i for i in content if str.isdigit(i)]
            xx=int(''.join(string1))
            guding.append(xx)
        path='./Ti/'+str(n+1)+'.bmp'
        section=cv2.imread(path,0)
        #print(guding)
        codelist=ExtractstrataSeqfromTI(section,codelist)

        #print guding[0],guding[1],guding[2],guding[3]
        m=sectionload_xG2(m,section,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale,guding[4]*scale,guding[5]*scale,jvalue)#载入剖面
        if flag==1:
            Ti,Tizuobiao=RecodeTIextendforEMG(section,m,template_x,template_y,guding[0]*scale,guding[1]*scale,guding[2]*scale,guding[3]*scale,guding[4]*scale,guding[5]*scale)
            Tilist.append(Ti)
            Tizuobiaolist.append(Tizuobiao)

    return m,Tilist,Tizuobiaolist,codelist


def initialAIforPythia(m, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, N, U, hardlist, code,
                       valueliata):
    # 全自动初始化流程整合,m为导入好剖面的待模拟网格
    # m为已经导入了Ti的模拟网格
    time_start1 = time.time()
    m = extendTimodel(m, template_h, template_x, template_y)  # 拓展模拟网格

    data = m.transpose(-1, -2, 0)  # 转置坐标系
    grid = tvtk.ImageData(spacing=(1, 1, -1), origin=(0, 0, 0),
                          dimensions=data.shape)
    grid.point_data.scalars = np.ravel(data, order='F')
    grid.point_data.scalars.name = 'lithology'
    write_data(grid, './output/outputinitial.vtk')
    np.save('./output/outputinitial.npy', m)
    print('extend done')

    my_file = Path("./database/Cdatabase.npy")
    if my_file.exists():
        Cdatabase = np.load('./database/Cdatabase.npy')
        cdatabase = np.load('./database/cdatabase.npy')
        database = np.load('./database/database.npy')
        zuobiaolist = np.load('./database/zuobiaolist.npy')
        print('Patterndatabase has been loaded!')
    else:
        print('Please wait for the patterndatabase building!')
        database, zuobiaolist = databasebuildAI(m, template_h, template_x, template_y)  # 数据库构建
        np.save('./database/database.npy', database)
        np.save('./database/zuobiaolist.npy', zuobiaolist)
        cdatabase = databasecataAI(database, lag)
        np.save('./database/cdatabase.npy', cdatabase)
        Cdatabase = databaseclusterAI(cdatabase, U)
        np.save('./database/Cdatabase.npy', Cdatabase)
        print('Patterndatabase has been builded!')
    time_end1 = time.time()
    print('数据库构建时间损耗:')
    print(time_end1 - time_start1)

    time_start = time.time()
    print('initial start:')
    #########################机器学习部分
    m = buildfaultkerasM3(m, valueliata, flaglist, epoch)
    np.save('./output/outputinitialFault.npy', m)

    #########################

    m = initialPythiasub(m, template_h, template_x, template_y, lag, lag_h, lag_x, lag_y, Cdatabase, cdatabase,
                         zuobiaolist, N, code, hardlist)

    time_end = time.time()

    np.save('./output/outputinitial3.npy', m)

    print('initial done')
    print('初始化建模时间损耗:')
    print(time_end - time_start)
    # 初始化
    return m


def extendTimodel(m, template_h, template_x, template_y):  # 全自动拓展插入硬数据的待模拟网格
    lag = max(template_h, template_x, template_y) // 2
    # lag=max(template_h,template_x,template_y)//2
    m2 = np.pad(m, lag, 'edge')
    d = []
    for h in range(lag, m2.shape[0] - lag):
        for x in range(lag, m2.shape[1] - lag):
            for y in range(lag, m2.shape[2] - lag):
                d.append((h, x, y))

    for cc in range(lag):
        # random.shuffle(d)
        flag = 0
        for n in range(len(d)):
            h = d[n][0]
            x = d[n][1]
            y = d[n][2]
            if m2[h, x, y] == -1:
                value = extend2dAI(m2, h, x, y)
                flag = -1
                if value != -1:
                    # print value
                    m[h - lag, x - lag, y - lag] = value

        if flag == 0:
            break
        m2 = np.pad(m, lag, 'edge')
        # 填充为1的
    return m


def simgridex(m, beilv):
    H0 = m.shape[0]
    W0 = m.shape[1]
    L0 = m.shape[2]

    dstHeight = int(H0 * beilv)
    dstWidth = int(W0 * beilv)
    dstLength = int(L0 * beilv)

    dst = np.zeros((dstHeight, dstWidth, dstLength), int)
    for i in range(0, dstHeight):
        for j in range(0, dstWidth):
            for k in range(0, dstLength):
                iNew = int(i * (H0 * 1.0 / dstHeight))
                jNew = int(j * (W0 * 1.0 / dstWidth))
                kNew = int(k * (L0 * 1.0 / dstLength))
                print(dst.shape, iNew, jNew, kNew)
                dst[i, j, k] = m[iNew, jNew, kNew]
    return dst


def sectionload_xG2(m, section, hz, hz2, xz, yz, xz2, yz2,
                    jivalue):  # 相对坐标的剖面导入,section为数组,xz,yz分别为剖面两端点的相对坐标,增加剖面上方赋值空值的功能
    # 斜剖面导入后需要扩充才正确 jivalue为基质值

    # 斜剖面导入后需要扩充才正确
    # print(hz,hz2)
    ns = section.shape[1]
    hc = float(hz2 - hz) + 1
    xc = float(xz2 - xz)
    yc = float(yz2 - yz)
    if xc < 0:
        xc1 = xc - 1
    else:
        xc1 = xc + 1
    if yc < 0:
        yc1 = yc - 1
    else:
        yc1 = yc + 1
    # 计量后加一为长度
    lv = int(max(abs(xc1), abs(yc1)))  # 比较长度绝对值大小，得到即为斜剖面需要填网格总数，所以需要加一
    xlv = xc / (lv - 1)
    ylv = yc / (lv - 1)
    x1 = xz
    y1 = yz
    # 对section的处理
    # h=m.shape[0]
    section = sectionexyunshi(section, int(hc), lv, jivalue)
    # print h
    # print section.shape[0],section.shape[1],lv,m.shape[0],m.shape[1],m.shape[2]
    for n in range(lv):
        m[hz:hz2 + 1, x1, y1] = section[:, n]
        # print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
        x1 = int(xz + (n + 1) * xlv + 0.5)  # 四舍五入
        y1 = int(yz + (n + 1) * ylv + 0.5)

    x1 = xz
    y1 = yz
    for n in range(lv):
        if section[0, n] == 0:
            m[0:hz + 1, x1, y1] = 0
            # print x1,y1,xz+(n*xlv),yz+(n*ylv)   检测用
        x1 = int(xz + (n + 1) * xlv + 0.5)  # 四舍五入
        y1 = int(yz + (n + 1) * ylv + 0.5)

    return m


def RecodeTIextendforEMG(section, m, template_x, template_y, h1, h2, x1, y1, x2, y2):  # EM迭代用剖面提取，(x,y）为剖面定位的一组坐标
    # m为已经完成扩展的模拟网格
    dx = []
    dy = []
    dh = []
    lag = max(template_x, template_y) // 2
    ms = -np.ones((m.shape[0], m.shape[1], m.shape[2]), int)
    sectionloadG(ms, section, h1, h2, x1, y1, x2, y2)  # 独立载入防止造成多剖面混乱
    Tizuobiaox = -np.ones((m.shape[0], m.shape[1], m.shape[2]), int)
    Tizuobiaoy = -np.ones((m.shape[0], m.shape[1], m.shape[2]), int)
    Tizuobiaoh = -np.ones((m.shape[0], m.shape[1], m.shape[2]), int)
    for h in range(Tizuobiaoh.shape[0]):
        for x in range(Tizuobiaoh.shape[1]):
            for y in range(Tizuobiaoh.shape[2]):
                Tizuobiaoh[h, x, y] = h
    if abs(h1 - h2) >= lag:
        for n1 in range(min(h1, h2), max(h1, h2) + 1):
            dh.append(n1)
    else:
        for n1 in range(max(0, min(h1, h2) - lag), min(max(h1, h2) + lag, m.shape[1] - 1) + 1):
            dh.append(n1)

    if abs(x1 - x2) >= lag:
        for n1 in range(min(x1, x2), max(x1, x2) + 1):
            dx.append(n1)
    else:
        for n1 in range(max(0, min(x1, x2) - lag), min(max(x1, x2) + lag, m.shape[1] - 1) + 1):
            dx.append(n1)

    if abs(y1 - y2) >= lag:
        for n1 in range(min(y1, y2), max(y1, y2) + 1):
            dy.append(n1)
    else:
        for n1 in range(max(0, min(y1, y2) - lag), min(max(y1, y2) + lag, m.shape[2] - 1) + 1):
            dy.append(n1)

    for h in range(ms.shape[0]):
        for x in range(ms.shape[1]):
            for y in range(ms.shape[2]):
                if ms[h, x, y] != -1:
                    Tizuobiaox[h, x, y] = x
                    Tizuobiaoy[h, x, y] = y
    temp = ms[:, dx, :]
    fowt = temp[:, :, dy]
    fow = fowt[dh, :, :]
    Tizuobiaoxt = Tizuobiaox[:, dx, :]
    Tizuobiaoxx = Tizuobiaoxt[:, :, dy]
    Tizuobiaox = Tizuobiaoxx[dh, :, :]
    Tizuobiaoyt = Tizuobiaoy[:, dx, :]
    Tizuobiaoyy = Tizuobiaoyt[:, :, dy]
    Tizuobiaoy = Tizuobiaoyy[dh, :, :]
    Tizuobiaoht = Tizuobiaoh[:, dx, :]
    Tizuobiaohh = Tizuobiaoht[:, :, dy]
    Tizuobiaoh = Tizuobiaohh[dh, :, :]
    c = max(fow.shape[1], fow.shape[2])
    # Tizuobiaoh=-np.ones((fow.shape[0],fow.shape[1],fow.shape[2]), int)

    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=extendTimodelsave, args=(fow, c, c, c, 1))

    p2 = multiprocessing.Process(target=extendTimodelsave, args=(Tizuobiaox, c, c, c, 2))
    p3 = multiprocessing.Process(target=extendTimodelsave, args=(Tizuobiaoy, c, c, c, 3))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()

    Tizuobiao = []  # 坐标矩阵
    Tizuobiaox = np.load('./output/ext2.npy')
    Tizuobiaoy = np.load('./output/ext3.npy')
    for h in range(Tizuobiaox.shape[0]):
        for x in range(Tizuobiaox.shape[1]):
            for y in range(Tizuobiaox.shape[2]):
                sodoi = np.array([Tizuobiaoh[h, x, y], Tizuobiaox[h, x, y], Tizuobiaoy[h, x, y]])
                Tizuobiao.append(sodoi)
    # Ti=extendTimodel(fow,c,c,c)
    Ti = np.load('./output/ext1.npy')
    return Ti, Tizuobiao