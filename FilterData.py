# -*- coding: UTF-8 -*-
# 对 user_action操作过滤
'''
# 将 user0 对 user1，2，3... 的 @ 转发 评论数 进行累加.。user num 的形式
mydict = {}
ratingList = []
pon = 0
pnum = 0
with open("Data/user_action.txt", "r")as cin:
    line = cin.readline()
    while line != None and line != "":
        arr = line.split("\t")
        user, user0, atNum, transmitNum, commentNum = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]), int(arr[4])
        # print  user, user0, atNum, transmitNum, commentNum
        num = atNum + transmitNum + commentNum
        if mydict.has_key(user):
            num = num + mydict.get(user)
        #     mydict[user] = num
        # else:
        #     mydict[user] = num
        mydict[user] = num
        line = cin.readline()
        pon=pon+1
        # print user, mydict[user]

print "总信息数：", pon
fco = open("Data/user_action01.txt", "w")
for i in mydict:
    print i, mydict[i]
    pnum = pnum+1
    fco.writelines(str(i)+"\t"+str(mydict[i])+"\n")
print "用户数：", len(mydict)
fco.close()



md={"a":29,"c":1,"b":3}
mg = sorted(md.items(), key = lambda item: item[1], reverse = True) #排序
print mg

'''
# 02 将 user0 对 user1，2，3... 的 @ 转发 评论数 进行累加.。user num 的形式  加排序
mydict = {}
ratingList = []
pon = 0
pnum = 0
with open("Data/user_action.txt", "r")as cin:
    line = cin.readline()
    while line != None and line != "":
        arr = line.split("\t")
        user, user0, atNum, transmitNum, commentNum = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]), int(arr[4])
        # print  user, user0, atNum, transmitNum, commentNum
        num = atNum + transmitNum + commentNum
        if mydict.has_key(user):
            num = num + mydict.get(user)
        #     mydict[user] = num
        # else:
        #     mydict[user] = num
        mydict[user] = num
        line = cin.readline()
        pon=pon+1
        # print user, mydict[user]
mg = sorted(mydict.items(), key = lambda item: item[1], reverse = True) #排序
"""
items()方法将字典的元素 转化为了元组，
而这里key参数对应的lambda表达式的意思则是选取元组中的第二个元素作为比较参数
（如果写作key=lambda item:item[0]的话则是选取第一个元素作为比较对象，也就是key值作为比较对象。
lambda x:y中x表示输出参数，y表示lambda 函数的返回值），
所以采用这种方法可以对字典的value进行排序。
！！！注意排序后的返回值是一个list，而原字典中的名值对被转换为了list中的元组！！！
"""
print "总信息数：", pon
fco = open("Data/user_action.txt", "w")
for i in mg:
    # print i, mydict[i]
    pnum = pnum+1
    fco.writelines(str(i[0])+"\t"+str(i[1])+"\n")
print "用户数：", len(mydict)
fco.close()

