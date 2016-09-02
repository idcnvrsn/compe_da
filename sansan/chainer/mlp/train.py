# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

#import data
import net


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from datetime import datetime
import math
from sklearn.metrics import mean_absolute_error

import os
import time

from time import clock
from PIL import Image
from PIL import ImageOps
from skimage.transform import resize

fMakeTrain = 0
fDoMode = 0

if fMakeTrain == 1:
    
    def load_data(file_name, img_dir, img_shape, orientations, pixels_per_cell, cells_per_block):
        classes = ['company_name', 'full_name', 'position_name', 'address', 'phone_number', 'fax', 'mobile', 'email', 'url']
        df = pd.read_csv(file_name)
        n = len(df)
        Y = np.zeros((n, len(classes)),dtype=np.int32)
        print('loading...')
        s = clock()
        for i, row in df.iterrows():
            f, l, t, r, b = row.filename, row.left, row.top, row.right, row.bottom
            print(i,f)
            img = Image.open(os.path.join(img_dir, f)).crop((l,t,r,b)) # 項目領域画像を切り出す
            if img.size[0]<img.size[1]:                                # 縦長の画像に関しては90度回転して横長の画像に統一する
                img = img.transpose(Image.ROTATE_90)
            
            # preprocess
            img = ImageOps.grayscale(img)

#            img_gray = img.convert('L')
#            img_gray = np.array(img_gray.resize(img_shape))/255.       # img_shapeに従った大きさにそろえる
            img_gray = np.array(img)
            img_gray =resize(img_gray,(50,300))
#            io.imsave("img"+str(i)+".bmp",img_gray)
    
            # feature extraction
#            img = np.array(hog(img_gray,orientations = orientations,
#                               pixels_per_cell = pixels_per_cell,
#                               cells_per_block = cells_per_block))   
            img_gray = img_gray.reshape(1,-1)
#            print(img_gray.shape)
            if i == 0:
#                feature_dim = len(img_gray)
                feature_dim = img_gray.shape[0] * img_gray.shape[1]
                print('feature dim:', feature_dim)
                X = np.zeros((n, feature_dim),dtype=np.float32)
                print(X.shape)
            
#            X[i,:] = np.array([img_gray])
            X[i,:] = img_gray.copy()
            y = list(row[classes])
            Y[i,:] = np.array(y)
        
        print('Done. Took', clock()-s, 'seconds.')
        return X, Y
        
    
    img_shape = (216,72)
    orientations = 6
    pixels_per_cell = (12,12)
    cells_per_block = (1, 1)
    X, y = load_data('../../train.csv', '../../train_images', img_shape, orientations, pixels_per_cell, cells_per_block)
#    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1234)
    joblib.dump(X,"X.pkl")
    joblib.dump(y,"y.pkl")
else:
    X = joblib.load("X.pkl")
    y = joblib.load("y.pkl")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=1234)
print('学習データの数:', len(X_train))
print('検証データの数:', len(X_test))

start = time.time()          

#構築したmodel群のリスト
models = []

batchsize = 100
n_epoch = 40
n_units = 1000

dir_name = datetime.now().strftime("%Y%m%d_%H%M%S") 
os.mkdir(dir_name)

#9クラス分の2分類器を学習と評価
for classe in range(0,9):
    #クラスに属する:1,属さない:0のデータで2分類器学習
    y_train_s = y_train[:,classe]
    y_test_s = y_test[:,classe]

    N_test = X_test.shape[0]
    
    N = X_train.shape[0]
    
    # Prepare multi-layer perceptron model, defined in net.py
    if args.net == 'simple':
        model = L.Classifier(net.MnistMLP(15000, n_units, 2))
        if args.gpu >= 0:
            cuda.get_device(args.gpu).use()
            model.to_gpu()
        xp = np if args.gpu < 0 else cuda.cupy
    elif args.net == 'parallel':
        cuda.check_cuda_available()
        model = L.Classifier(net.MnistMLPParallel(15000, n_units, 2))
        xp = cuda.cupy
    
    # Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    # Init/Resume
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_npz(args.initmodel, model)
    if args.resume:
        print('Load optimizer state from', args.resume)
        serializers.load_npz(args.resume, optimizer)
    
    # Learning loop
    for epoch in six.moves.range(1, n_epoch + 1):
        print('epoch', epoch)
    
        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N, batchsize):
            x = chainer.Variable(xp.asarray(X_train[perm[i:i + batchsize]]))
            t = chainer.Variable(xp.asarray(y_train_s[perm[i:i + batchsize]]))
    
            # Pass the loss function (Classifier defines it) and its arguments
            optimizer.update(model, x, t)
    
            if epoch == 1 and i == 0:
                with open('graph' + str(classe) + '.dot', 'w') as o:
                    g = computational_graph.build_computational_graph(
                        (model.loss, ), remove_split=True)
                    o.write(g.dump())
                print('graph' + str(classe) + ' generated')
    
            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
    
        print('train' + str(classe) + ' mean loss={}, accuracy={}'.format(
            sum_loss / N, sum_accuracy / N))
    
        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N_test, batchsize):
            x = chainer.Variable(xp.asarray(X_test[i:i + batchsize]),
                                 volatile='on')
            t = chainer.Variable(xp.asarray(y_test_s[i:i + batchsize]),
                                 volatile='on')
            loss = model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
    
        print('test' + str(classe) + '  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))

    models.append(model)
    
    # Save the model and the optimizer
    print('save the model' + str(classe))
    serializers.save_npz(dir_name + "/" + 'mlp' + str(classe) + '.model', model)
    print('save the optimizer' + str(classe))
    serializers.save_npz(dir_name + "/" + 'mlp' + str(classe) + '.state', optimizer)

#９個の分類器をテストデータの各行に適用
pred = np.zeros(y_test.shape,dtype=y_test.dtype)
for i_cls in range(9):
    print(i_cls)
    #予測
    
    xp = np
    x = chainer.Variable(xp.asarray(X_test))
    
    pred_raw = models[i_cls].to_cpu().predictor(x)
#        np.max(pred.data,axis=1)
    
    pred_bin = np.argmax(pred_raw.data,axis=1)
    
    pred[:,i_cls] = pred_bin

elapsed_time = time.time() - start
print( ("elapsed_time:{0}".format(elapsed_time)) + "[sec]")

print('test score',accuracy_score(y_test,pred))
print(classification_report(y_test, pred))

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))
print('MAE:', mae(y_test, pred))

logfilename = dir_name + '/' + 'log.txt'
with open(logfilename, "w") as file:
    file.write(("elapsed_time:{0}".format(elapsed_time)) + "[sec]\n")
    file.write("epoch:" + str(n_epoch))
    file.write("train sample num:" + str(X_train.shape[0]) + '\n')
    file.write('\n')
    file.write('test accuracy score:' + str(accuracy_score(y_test,pred)) + '\n')
    file.write('test MAE:' + str(mae(y_test, pred)) + '\n\n')
    file.write(str(classification_report(y_test, pred))  + '\n')
#        file.write(str(confusion_matrix(y_test, pred)))