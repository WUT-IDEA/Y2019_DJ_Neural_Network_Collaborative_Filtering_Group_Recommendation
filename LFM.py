#-*- coding: UTF-8 -*-
import numpy as np
import math
import operator
import time
import random

def initpara(users, items, F):
    p = dict()
    q = dict()

    for userid in users:
        p[userid] = [(-1 + 2 * random.random()) for f in range(0, F)]  # / math.sqrt(F)

    for itemid in items:
        q[itemid] = [(-1 + 2 * random.random()) for f in range(0, F)]  # / math.sqrt(F)

    return p, q

def initsamples(user_items):
    user_samples = []
    items_pool = []
    for userid, items in user_items.items():
        for item in items:
            items_pool.append(item)

    for userid, items in user_items.items():
        samples = dict()
        for itemid, score in items.items():
            if score != 0:
                samples[itemid] = score
        user_samples.append((userid, samples))

    return user_samples

def initmodel(user_items, users, items, F):
    p, q = initpara(users, items, F)
    user_samples = initsamples(user_items)

    return p, q, user_samples

def predict(userid, itemid, p, q):
    a = sum(p[userid][f] * q[itemid][f] for f in range(0, len(p[userid])))
    return a


def lfm(user_items, users, items, F, N, alpha, lamda):
    '''
    LFM计算参数 p,q
    :param user_items: user_items
    :param users: users
    :param items: items
    :param F: 隐类因子个数
    :param N: 迭代次数
    :param alpha: 步长
    :param lamda: 正则化参数
    :return: p,q
    '''
    p, q, user_samples = initmodel(user_items, users, items, F)

    debugid1 = 0
    debugid2 = 0
    for step in range(0, N):
        random.shuffle(user_samples)  # 乱序

        error = 0
        count = 0
        for userid, samples in user_samples:
            for itemid, rui in samples.items():
                pui = predict(userid, itemid, p, q)
                eui = rui - pui
                count += 1
                error += math.pow(eui, 2)
                '''debug'''
                if userid == 1:
                    if debugid1 == 0 and rui == 1:
                        debugid1 = itemid
                    if debugid2 == 0 and rui == -1:
                        debugid2 = itemid

                if userid == 1 and itemid == debugid1:
                    print debugid1, rui, pui, eui, alpha
                if userid == 1 and itemid == debugid2:
                    print debugid2, rui, pui, eui, alpha

                '''debug end'''

                for f in range(0, F):
                    p_u = p[userid][f]
                    q_i = q[itemid][f]

                    p[userid][f] += alpha * (eui * q_i - lamda * p_u)
                    q[itemid][f] += alpha * (eui * p_u - lamda * q_i)

        rmse = math.sqrt(error / count)
        print  "rmse:", rmse
        alpha *= 0.9
    return p, q


def predictlist(userid, items, p, q):
    predict_score = dict()
    for itemid in items:
        p_score = predict(userid, itemid, p, q)
        predict_score[itemid] = p_score

    return predict_score


def recommend():
    print 'start'
    user_items = {1: {'a': 1, 'b': -1, 'c': -1, 'd': -1, 'e': 1, 'f': 1, 'g': -1},
                  2: {'a': -1, 'b': 1, 'c': -1, 'd': 1, 'e': 1, 'f': 1, 'g': 1},
                  3: {'a': 1, 'b': -1, 'c': 0, 'd': -1, 'e': -1, 'f': -1, 'g': 1},
                  4: {'a': 1, 'b': -1, 'c': -1, 'd': 0, 'e': 1, 'f': 1, 'g': 1},
                  5: {'a': -1, 'b': 1, 'c': 1, 'd': 1, 'e': -1, 'f': -1, 'g': 0},
                  6: {'a': 1, 'b': 0, 'c': -1, 'd': -1, 'e': 1, 'f': -1, 'g': -1}}
    users = {1, 2, 3, 4, 5, 6}
    items = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
    F = 5
    N = 30
    alpha = 0.3
    lamda = 0.03
    p, q = lfm(user_items, users, items, F, N, alpha, lamda)

    for userid, itemdic in user_items.items():
        print userid
        print "target", itemdic
        predict_score = predictlist(userid, itemdic, p, q)
        print  "predicted", predict_score

    print 'end'

def getMLData():  # 获取训练集和测试集的函数
    import re
    f = open("Data/arec.train.rating", 'r')
    lines = f.readlines()
    f.close()
    data = []
    for line in lines:
        list = re.split('\t|\n', line)
        if int(list[2]) != 0:
            data.append([int(i) for i in list[:3]])
    train_data = data
    f = open("Data/arec.test.rating", 'r')
    lines = f.readlines()
    f.close()
    data = []
    for line in lines:
        list = re.split('\t|\n', line)
        if int(list[2]) != 0:
            data.append([int(i) for i in list[:3]])
    test_data = data

    return train_data, test_data

t1 = time()
train_data, test_data = getMLData()
t2 = time()
print('model time T2 = =  [%.1f s]' % (t2 - t1))
a.train()
t3 = time()
print('train time T3 = =  [%.1f s]' % (t3 - t2))
a.test(test_data)
print('test time    T4 = =  [%.1f s]' % (time() - t3))
