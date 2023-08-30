######## semantic operation  #####


######   从TI中提取地层层序
### 输入：Ti----  TI
###      strSeq  -----  空
###  输出： strSeq  ----  返回地层层序

def ExtractstrataSeqfromTI(Ti,strSeq):#提取TI地层层序
    for x in range(Ti.shape[1]):
        TempStrSeq=[]
        TempStrSeq.append(Ti[0,x])
        for h in range(1,Ti.shape[0]):
            if Ti[h,x]!=Ti[h-1,x]:
               if Ti[h,x]!=-1:
                  TempStrSeq.append(Ti[h,x])
        if TempStrSeq not in strSeq:
           strSeq.append(TempStrSeq)
    return strSeq