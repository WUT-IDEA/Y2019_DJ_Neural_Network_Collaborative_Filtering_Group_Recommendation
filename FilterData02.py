# -*- coding: UTF-8 -*-
# 对 rec_log 排 序

# 04 # 对 rec_log 排 序
# 根据用户 对 其他用户的 @ 转发 评论数 进行累加和，排序，过滤总数小于180 的用户
# 过滤 并排序
import time
start = time.time()

ucin = open("Data/rec_log_train_rating.txt", "r")
uline = ucin.readline().strip("\n")  #''' 因为最后一行结束的时候，没有换行符\n，所以导致写的数据中，有一行是两行,所以都去掉换行 '''
userList = []
print "打开了","\n"
while uline is not None and uline != "":
    ab = uline.split("\t", 1)
    user0, other = int(ab[0]), ab[1]
    userList.append((user0, other))
    uline = ucin.readline().strip("\n")  #''' 因为最后一行结束的时候，没有换行符\n，所以导致写的数据中，有一行是两行,所以都去掉换行 '''
ucin.close()
end = time.time()
print end-start
print "读完了\n"

userList = sorted(userList, key = lambda item: item[0])  #排序
end = time.time()
print end-start

print "开始排序\n"
# print userList
ucout = open('Data/rec_log_train_rating.rating', 'w')
sign = True
for item in userList:  #最后一行 不要换行
    if sign:
        ucout.write(str(item[0]) + '\t' + item[1])
        sign = False
    else:
        ucout.write('\n' + str(item[0]) + '\t' + item[1])

end = time.time()
print end-start
ucout.close()
print "排序了\n"