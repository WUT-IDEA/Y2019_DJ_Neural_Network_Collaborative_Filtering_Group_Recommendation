# -*- coding: UTF-8 -*-

# #读取文件  训练文件里面 没有操作的，即值不是1（接受），-1（不接受），而是0的就没有价值
# #  rec_log.train.rating         rec_logw.test.rating  中不存在有不接受的
# # 把rec_log.test.rating 写进rec_logw.test.rating 中
# ratingList = []
# pon = 0
# nen = 0
# with open("Data/user_action", "r")as cin:
#     line = cin.readline()
#     while line != None and line != "":
#         arr = line.split("\t")
#         user, item, rate, datast = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
#         if rate <> 0:
#             ratingList.append(line)
#             # ratingList.append(str(user + '\t' + item + '\t' + rate + '\t' + datast + "\n"))
#             pon += 1
#             # print user, item, rate
#         else:
#             nen += 1
#         line = cin.readline()
# # arr = str("l"+'\t'+"ine\n"+"2")
# # ratingList.append(arr)
# # print arr
# # print ratingList
# print "总信息数：", pon+nen, "\n 有效信息数：", pon,"\n 无效信息数：", nen,
# fco = open("Data/rec_logw.test.rating", "w")
# fco.writelines(ratingList)
# fco.close()
# 读取文件  训练文件里面 没有操作的，即值不是1（接受），-1（不接受），而是0的就没有价值
#  rec_log.train.rating         rec_logw.test.rating  中不存在有不接受的
# 把rec_log.test.rating 写进rec_logw.test.rating 中
'''
ratingList = []
pon = 0
nen = 0
with open("Data/rec_log.test.rating", "r")as cin:
    line = cin.readline()
    while line != None and line != "":
        arr = line.split("\t")
        user, item, rate, datast = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
        if rate != 0:
            # ratingList.append(str(user,item) + '\t' + rate + '\t' + datast + "\n")
            pon += 1
            # print user, item, rate
        else:
            nen += 1
        line = cin.readline()
# arr = str("l"+'\t'+"ine\n"+"2")
# ratingList.append(arr)
# print arr
# print ratingList
print "总信息数：", pon+nen, "\n 有效信息数：", pon,"\n 无效信息数：", nen,
fco = open("Data/rec_logw.test.rating", "w")
fco.writelines(ratingList)
fco.close()
'''

# 导入相应的库（对数据库进行切分需要用到的库是sklearn.model_selection 中的 train_test_split）
import numpy as np

from sklearn.model_selection import train_test_split

# 首先，读取文件成矩阵的形式。
my_matrix = np.loadtxt("Data/rec_log_new.rating", dtype=int, delimiter="\t", skiprows=0)
# print my_matrix
# 对于矩阵而言，将矩阵倒数第一列之前的数值给了X（输入数据），将矩阵大最后一列的数值给了y（标签）
X, y = my_matrix[:, :-1], my_matrix[:, -1]

# 训练集：测试集=7:3的
# 概率划分，到此步骤，可以直接对数据进行处理，这个鬼函数需要将矩阵划分两部分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 此步骤，是为了将训练集与数据集的数据合并


# np.column_stack将两个矩阵进行组合连接
train = np.column_stack((X_train, y_train))
# numpy.savetxt 将txt文件保存到文件, fmt="%d"为整型，否则默认为float
np.savetxt('Data/arec.train.rating', train, fmt="%d", delimiter='\t')

test = np.column_stack((X_test, y_test))
np.savetxt('Data/arec.test.rating', test, fmt="%d", delimiter='\t')
