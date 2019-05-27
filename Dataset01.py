# -*- coding: UTF-8 -*-
import scipy.sparse as sp
import numpy as np

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testMatrix = self.load_rating_file_as_matrix(path + ".test.rating")
        self.testList, self.scoreList = self.load_rating_file_as_list(path + ".test.rating")

        self.num_users = 61083
        self.num_items = 4497
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        scoreList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                ratingList.append( np.array([user, item]) )
                scoreList.append(rating)
                line = f.readline()
        return ratingList, scoreList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Construct matrix 构造矩阵
        num_users = 61083
        num_items = 4497
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                mat[user, item] = rating
                line = f.readline()    
        return mat