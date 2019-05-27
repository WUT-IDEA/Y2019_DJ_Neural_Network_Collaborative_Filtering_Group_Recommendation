#-*- coding: UTF-8 -*-
# user_num 为所分群组的成员个数,为了方便计算，一般设为3，先将payoff矩阵变为多维矩阵，payoff矩阵的值为 元组
# item_num为取每个成员最感兴趣的item的个数.另外取一矩阵记录对应itemid。
user_num = 3
item_num = 4

# 判断这个节点是否是纳什均衡点，节点值是 元组
def judge(o,i,j):
    for jo in item_num:
        if( pay_matrix[jo][i][j] > pay_matrix[o][i][j] ): return False
    for ji in item_num:
        if( pay_matrix[o][ji][j] > pay_matrix[o][i][j] ):return False
    for jj in item_num:
        if( pay_matrix[o][i][jj] > pay_matrix[o][i][j] ):return False
    return True

# 多个纳什均衡点， 通过调和均值函数，求最佳均衡点，节点值是 元组
def harmonic_mean(nash_list):
    result_list = []
    asult = 0
    for item in nash_list:
        for x in item[0:3]:
            asult = asult + 1.0/x
        result_list.append(asult)
    inde = result_list.index(min(result_list))
    return nash_list[inde]

# 构建 item_num^user_num矩阵
# result 格式为
# [,],[,]...
# ......
# [,],[,]...
file = open("result.txt","r")
list = file.readlines()
pay_matrix = []
m0=[]
# 记录纳什均衡点，最后比较取最优
nash_list=[]
for aa in list:
    aa=aa.strip()
    aa = aa.split("],[")
    for a in aa:
        a.strip("[");
        a.strip("]");
        a = a.split(",");
        m0.append(a);
    pay_matrix.append(m0);

for o in range(item_num):
    for i in range(item_num):
        for j in range(item_num):
            res = judge(o,i,j)
            if (res):
                nash_item = pay_matrix[o][i][j].append(o)
                nash_item = pay_matrix[o][i][j].append(i)
                nash_item = pay_matrix[o][i][j].append(j)
                nash_list.append(nash_item)
en = len(nash_list)
# 最终输出的元组，前一半是纳什均衡预测评分值，后一半是对应的多维数组的index从而寻找对应的item编号。
if en ==0:
    print "null"
elif en ==1:
    print nash_list[0]
else:
    print harmonic_mean(nash_list)
