# -*- coding: UTF-8 -*-

import numpy as np
import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l1, l2
from keras.models import Sequential, Model, model_from_json
from keras.layers import Embedding, Input, merge
from keras.layers.core import Dense, Flatten
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.utils import np_utils
from Dataset01 import Dataset
import h5py
from evaluate01 import evaluate_model
from Dataset01 import Dataset
from time import time
import sys
import LFM, MLP
import scipy.sparse as sp

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in xrange(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])
    return model

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    # assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  init=init_normal, W_regularizer=l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  init=init_normal, W_regularizer=l2(reg_mf), input_length=1)
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] / 2, name="mlp_embedding_user",
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] / 2, name='mlp_embedding_item',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = merge([mf_user_latent, mf_item_latent], mode='mul')  # element-wise multiply
    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode='concat')
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)
    # Concatenate MF and MLP parts
    # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = merge([mf_vector, mlp_vector], mode='concat')
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name="prediction")(predict_vector)
    model = Model(input=[user_input, item_input],
                  output=prediction)
    return model

def load_rating_file_as_matrix(filename):
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    # Construct matrix 类型转换list to arry
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if (rating > 0):
                mat[user, item] = 1.0
            line = f.readline()
    return mat

def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            # print arr
            user, item = int(arr[0]), int(arr[1])
            ratingList.append((user, item))
            line = f.readline()
    return ratingList

# #读取文件  训练文件里面 没有操作的，即值不是1（接受），-1（不接受），而是0的就没有价值
# #  rec_log.train.rating         rec_logw.test.rating  中不存在有不接受的
# ratingList = []
# pon = 0
# nen = 0
# with open("Data/rec_log.test.rating", "r")as cin:
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
# #类型转换list to arry
# testRatings = load_rating_file_as_list("Data/ml-1m.test.rating")
# # print train, "!!!!\n!!!!\n", testRatings
#
# num_users, num_items = train.shape
# mf_dim = 8
# layers = [64,32,16,8]
# reg_mf = 0.0
# reg_layers = [0,0,0,0]
# learner = 'adam'
# lfm_pretrain = ''
# mlp_pretrain = ''
# learning_rate = 0.001
# test = load_rating_file_as_matrix("Data/ml-1m.test.rating")
# print test
# print testRatings[0]
#
# # Build model
# model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
# if learner.lower() == "adagrad":
#     model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
# elif learner.lower() == "rmsprop":
#     model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
# elif learner.lower() == "adam":
#     model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
# else:
#     model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
#
# # Load pretrain model
# if lfm_pretrain != '' and mlp_pretrain != '':
#     gmf_model = GMF.get_model(num_users, num_items, mf_dim)
#     gmf_model.load_weights(lfm_pretrain)
#     mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
#     mlp_model.load_weights(mlp_pretrain)
#     model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
#     print("Load pretrained GMF (%s) and MLP (%s) models done. " % (lfm_pretrain, mlp_pretrain))
#
# model.load_weights("Pretrain/ml-1m_NeuMF_8_[64,32,16,8]_1526895093.h5")
# re = model.evaluate(test, testRatings, batch_size=128, verbose=1, sample_weight=None)
# print re[0], re[1]

