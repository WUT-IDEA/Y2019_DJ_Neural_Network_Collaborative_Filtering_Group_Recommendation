# -*- coding: UTF-8 -*-
# 对 rec_log 排 序

# 05 # rec_log_train_rating.rating中 用户 和 物品 编号不连续，这导致 直接以两编号创建矩阵的方式太浪费空间
# 将两者编号，重新排序
import time
start = time.time()

icin = open("Data/item.rating", "r")

itemList = []
print "打开了","\n"
iline = icin.readline()
while iline is not None and iline != "":
    ite = iline.split("\n", 1)
    item = int(ite[0])
    itemList.append(item)
    iline = icin.readline()
icin.close()
end = time.time()
print end-start
print "s ,读完了item\n"

#常数，记载user是否是上一个user
user000 = -1
uNum = -1
ucout = open('Data/rec_log_new.rating', 'w')

lineList = []
rltNum = 0
with open("Data/rec_log_train_rating.rating", "r")as ucin:
    for line in ucin:
        arr = line.split("\t")
        user,item,rate = int(arr[0]), int(arr[1]), int(arr[2])
        if item == 1000359:
            print "find item0!!!!!!!!!!!!!!!!!!!!!"+str(rltNum)
        if user != user000:
            user000 = user
            uNum += 1
        user = uNum
        item = itemList.index(item)
        if rltNum == 0:
            line = str(user) + '\t' + str(item) + '\t' + str(rate)
        else:
            line = '\n'+str(user)+'\t' + str(item) + '\t' + str(rate)
        rltNum += 1
        lineList.append(line)
        # ucout.write(str(user)+'\t' + str(item) + '\t' + str(rate))
        """ 做一循环，满一万次存一次数据 """
        if rltNum % 10000 == 0:
            print str(rltNum)+'   '+str(user)
            ucout.writelines(lineList)
            lineList = []
if rltNum % 10000 != 0:
    ucout.writelines(lineList)  # 当最后一截，不满一万的时候，不能漏了


end = time.time()
print end-start
ucout.close()
print "改完\n"