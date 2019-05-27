# -*- coding: UTF-8 -*-
import numpy as np
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from Dataset01 import Dataset
from evaluate01 import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
from keras import backend

# def rmse(y_true, y_pred):
#     return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, latent_dim, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  init = init_normal, W_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  init = init_normal, W_regularizer = l2(regs[1]), input_length=1)   
    
    # Crucial to flatten an embedding vector! 降维 为潜向量
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    
    # Element-wise product of user and item embeddings  用户和项目  点积（Element-wise product） 的潜入层
    predict_vector = merge([user_latent, item_latent], mode = 'mul')
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    #  Dense(标准的一维全连接层)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(predict_vector)
    
    model = Model(input=[user_input, item_input], 
                output=prediction)

    return model

def get_train_instances(train):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(train[u, i]) # user对item的打分值
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors # 神经元数
    regs = eval(args.regs) # 用户和项目嵌入层的初始化
    # num_negatives = args.num_neg # 选择负面反馈数据的样本数
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs # 训练次数
    batch_size = args.batch_size  #小批次GD 样本的数量
    verbose = args.verbose # 显示每次训练的性能
    topK = 10
    evaluation_threads = 1  #线程数
    print("GMF01 arguments: %s" %(args))
    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5' %(args.dataset, num_factors, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testMatrix = dataset.trainMatrix, dataset.testMatrix
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testMatrix)))
    
    # Build model
    model = get_model(num_users, num_items, num_factors, regs)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')#, metrics=[rmse])
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')#, metrics=[rmse])
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')#, metrics=[rmse])
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')#, metrics=[rmse])
    #print(model.summary())
    
    # Init performance
    t1 = time()
    rmses = evaluate_model(model, testMatrix)
    rmse = np.array(rmses).mean()

    print('Init: RMSE = = %.5f\t [%.1f s]' % (rmse, time()-t1))
    
    # Train model
    best_rmse, best_iter = rmse, -1
    for epoch in xrange(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation 评估MLP.py
        if epoch %verbose == 0:
            rmses = evaluate_model(model, testMatrix)
            rmse, loss = np.array(rmses).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: RMSE = = %.5f, loss = %.4f [%.1f s]'
                  % (epoch,  t2-t1, rmse, loss, time()-t2))
            if rmse < best_rmse:
                best_iter, best_rmse =  epoch, rmse
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  RMSE = = %.5f.)" %(best_iter, best_rmse))
    if args.out > 0:
        print("The best LFM model is saved to %s" %(model_out_file))

# python GMF01.py --dataset arec --epochs 20 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1