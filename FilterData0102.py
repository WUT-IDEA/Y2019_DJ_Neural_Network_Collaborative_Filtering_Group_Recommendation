# -*- coding: UTF-8 -*-

# 04 过滤出rec_log_train中的商品记录数。
# 同上，

item_file = open('Data/item.txt', 'w')  # 只加入不覆盖
itemList = []  # 捕获item数
rltNum = 0

print "打开原始数据集\n"
flag = True
with open("Data/rec_log_train_rating.rating", "r")as cin:
    for line in cin:
        # # 对于中间中断的数据，找到了继续查
        # if flag and line != "1678533	647356	-1	1320547090\n":
        #     continue
        # elif flag:
        #     flag = False
        #     print "找到了，继续\n"
        #     continue
        arr = line.split("\t")
        item = str(arr[1])+'\n'
        if item in itemList:
            continue
        rltNum = rltNum + 1
        print rltNum
        itemList.append(item)

    item_file.writelines(itemList)  # 当最后一截，不满一万的时候，不能漏了

## 对列表排序
ucout = open('Data/item.rating', 'w')
itemList.sort()
ucout.writelines(itemList)

print "有效记录数", rltNum
item_file.close()

