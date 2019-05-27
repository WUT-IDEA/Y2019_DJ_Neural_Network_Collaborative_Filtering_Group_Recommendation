#-*- coding: UTF-8 -*-
# #这是python 2 下面写的，用的raw_input
#打开输入的文件
'''
u = "2088948	1760350	-1	1318348785"
u1,u2,u3 = u.split('	',2)   # '	'不同于' '
print u1,u2
u11=int(u1) #强制类型转换
u21=int(u2)
u4 = '(%d,%d) %d d' % (u11,u21,
                       1)
print u4
'''
old_file = open('Data/rec_log.train.rating','r')
print "open1"
new_file0 = open('Data/rec.train.rating','w')#打开新文件，因为不存在，用‘w’命名生成
new_file1 = open('Data/rec.test.rating','w')#打开新文件，因为不存在，用‘w’命名生成
new_file2 = open('Data/rec.test.negative','w')#打开新文件，因为不存在，用‘w’命名生成

print "open2"
#循环，一次读取旧文件的一行，直至content=0 也就是没内容了
i=0
content = old_file.readline()#读取一行
print content
while content != 0:
     if i%3 == 0:
          #数据集特点，每个用户三行评分，正样本测试集。取第一个为正样本测试集，后两个训练
          new_file1.write(content)
          u11,u12,u13 = content.split('	', 2) # 读取一行的前两个
          u1=int(u11)
          u2=int(u12)
          u3 = '(%d,%d) %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n' % (u1,u2,(u2+1)%1000,(u2+12)%1000,(u2+23)%1000,(u2+34)%1000,(u2+45)%1000,(u2+56)%1000,(u2+67)%1000,(u2+78)%1000,(u2+89)%1000,(u2+90)%1000,(u2+71)%1000,(u2+62)%1000,(u2+53)%1000,(u2+44)%1000,(u2+97)%1000,(u2+83)%1000,(u2+9831)%10000)
          new_file2.write(u3)
     else:
          new_file0.write(content)
     content = old_file.readline()
     i+=1
 #别忘了关闭文件
old_file.close()
new_file0.close()
new_file1.close()
new_file2.close()
