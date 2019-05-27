# -*- coding: UTF-8 -*-
import linecache


# # 快速读取文件的行数
# count = 0
# for index, line in enumerate(open("Data/rec_log_train_rating.txt",'r')):
#     count += 1
# print count


# # #快速读取某行的内容
# cou = linecache.getline("Data/rec_log_train_rating.txt",150000)
# print cou


# # 找出rec_log_train_rating.txt末的元素，  在rec_log_train.txt的行数
# flag = True
# num = 0
# with open("Data/rec_log_train.txt", "r")as cin:
#     for line in cin:
#         num = num+1
#         if flag and line != "381351	1760350	-1	1318419315\n":
#             continue
#         else:
#             print num, " ", line
#             flag = False
#             break


# # #快速读取rec_log_train_rating某行的内容，并查找在在rec_log_train的行数
cou = linecache.getline("Data/rec_log_train_rating.txt",5990000)
# print cou
flag = True
num = 0
with open("Data/rec_log_train.txt", "r")as cin:
    for line in cin:
        num = num+1
        if flag and line != cou:
            continue
        else:
            print num, " \n", line
            flag = False
            break

