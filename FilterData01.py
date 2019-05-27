# -*- coding: UTF-8 -*-
# 对 user_action操作过滤

# 03 过滤出rec_log_train中的活跃用户记录。
# 根据用户 对 其他用户的 @ 转发 评论数 进行累加和，排序，总数小于6 的用户过滤

itemList = []  # 捕获item数
# itemList_old = []  # 捕获item总数，与itemList对比
userList = []  # 过滤表，不在此列表的数据不要

uco = open("Data/user_action02.txt", "r")
aline = uco.readline()
while aline is not None and aline != "":
    ab = aline.split("\t")
    user0, anum = int(ab[0]), int(ab[1])
    # print user0
    if anum <= 180:  #总数小于180的用户过滤
        break
    userList.append(user0)
    aline = uco.readline()
uco.close()
print "读完了用户列表\n"

# itemco = open("Data/user_action02.txt", "r")
# iline = itemco.readline()
# while iline is not None and iline != "":
#     ib = iline.split("\t")
#     item0 = int(ib[0])
#     itemList.append(item0)
#     iline = itemco.readline()
# itemco.close()
# print "读完了商品列表\n"

print "打开原始数据集\n"
# item_file = open('Data/item.txt', 'a')  # 只加入不覆盖
# item_file_old = open('Data/item_old.txt','w')  #可能存在不活跃用户才评论过的项目

'''
#   'w' 重新开始
new_file = open('Data/rec_log_train_rating.txt', 'w')  #   'w' 重新开始
rltNum = 0
'''
 #   'a' 只加入不覆盖
new_file = open('Data/rec_log_train_rating.txt', 'a')  #   'a' 只加入不覆盖
rltNum = 690000


lineList = []
flag = True
with open("Data/rec_log_train.txt", "r")as cin:
    for line in cin:
        # # 对于中间中断的数据，找到了继续查
        # if flag and line != "1678533	647356	-1	1320547090\n":
        #     continue
        # elif flag:
        #     flag = False
        #     print "找到了，继续\n"
        #     continue
        arr = line.split("\t")
        user = int(arr[0])
        # user, item = int(arr[0]), int(arr[1])
        if user not in userList:
            continue
        rltNum = rltNum + 1
        lineList.append(line)
        # if item not in itemList:
        #     itemList.append(item)
        """ 做一循环，满一万次存一次数据 """
        if rltNum % 10000 == 0:
            print rltNum
            new_file.writelines(lineList)
            lineList = []



# with open("Data/rec_log_train.txt", "r")as cin:
#     line = cin.readline()  # 循环  进一 的标志
#     while line is not None and line != "" and line != "\t":
#         arr = line.split("\t")
#         user, item = int(arr[0]), int(arr[1])
#         # print num
#         # num=num+1
#         # if item not in itemList_old:
#         #     itemList_old.append(item)
#         if user not in userList:
#             line = cin.readline()  # 循环  进一 的标志
#             continue
#         if item not in itemList:
#             itemList.append(item)
#         if rltNum % 10000 == 0 and rltNum > 1:
#             print rltNum
#             new_file.writelines(lineList)
#             lineList = []
#         lineList.append(line)
#         rltNum = rltNum + 1
#         # print line
#         line = cin.readline()  # 循环  进一 的标志

if rltNum % 10000 != 0:
    new_file.writelines(lineList)  # 当最后一截，不满一万的时候，不能漏了

print "有效记录数", rltNum
# item_file.writelines(itemList)
# item_file_old.write(itemList_old)
# print " , 商品总数：  ", len(itemList)  # ," , 真正商品总数：  ",len(itemList_old)
# 别忘了关闭文件

new_file.close()
# item_file.close()
# item_file_old.close()
