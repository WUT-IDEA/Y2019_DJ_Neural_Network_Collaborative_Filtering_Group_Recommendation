# -*- coding: UTF-8 -*-
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testMatrix = None

def evaluate_model(model, testMatrix):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.

           trainMatrix  训练矩阵4列
                 testMatrixs  测试矩阵2列
    """
    global _model
    global _testMatrix
    global _testList
    # global _K
    _model = model
    _testMatrix = testMatrix
    # hits, ndcgs = [], []
    # if(num_thread > 1): # Multi-thread 多线程
    #     pool = multiprocessing.Pool(processes=num_thread)
    #     res = pool.map(eval_one_rating, range(len(_testMatrixs)))
    #     pool.close()
    #     pool.join()
    #     hits = [r[0] for r in res]
    #     ndcgs = [r[1] for r in res]
    #     rmses = [r[1] for r in res]
    #     return (hits, ndcgs, rmses)
    user_input, item_input, labels = get_test_instances(testMatrix)

    errors = []
    # Single thread 单线程
    predictions = _model.predict([np.array(user_input), np.array(item_input)], verbose=0)

    for idx in range(len(labels)):
        error = labels[idx] - predictions[idx,0]  #误差值
        squaredError = error * error   # target-prediction之差平方
        errors.append(squaredError)

    rmses =  math.sqrt(sum(errors) / len(errors))  # 均方根误差RMSE
    return rmses

def get_test_instances(testMatrix):
    user_input, item_input, labels = [],[],[]
    num_users = testMatrix.shape[0]
    for (u, i) in testMatrix.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(testMatrix[u, i]) # user对item的打分值
    # print "~~~~labels ~~~~~   \n"
    # print labels.length
    return user_input, item_input, labels