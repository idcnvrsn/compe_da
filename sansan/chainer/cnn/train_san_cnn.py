# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:15:47 2016
"""
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
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

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import six

fMakeTrain = 1
fDoMode = 0

if fMakeTrain == 1:
    # 予測出力ファイル
    fn_prediction = 'prediction.csv'
    
    # ダウンロードデータ
    train_csv    = '../../train.csv'
    train_images = '../../train_images'
    test_csv     = 'test.csv'
    test_images  = 'test_images'
    
    _columns = ['company_name', 'full_name', 'position_name', 'address', 'phone_number', 'fax', 'mobile', 'email', 'url']
    
    _img_len = 96
    
    _batch_size = 128
    _nb_epoch   = 20
    _sgd_lr     = 0.1
    _sgd_decay  = 0.001
    _Wreg_l2    = 0.0001
    
    def _load_rawdata(df, dir_images):
        '''画像を読み込み、4Dテンソル (len(df), 1, _img_len, _img_len) として返す
        '''
    
        X = np.zeros((len(df), 1, _img_len, _img_len), dtype=np.float32)
    
        for i, row in df.iterrows():
            print(i,dir_images, row.filename)
            img = Image.open(os.path.join(dir_images, row.filename))
            img = img.crop((row.left, row.top, row.right, row.bottom))
            img = img.convert('L')
            img = img.resize((_img_len, _img_len), resample=Image.BICUBIC)
    
            # 白黒反転しつつ最大値1最小値0のfloat32に画素値を正規化
            img = np.asarray(img, dtype=np.float32)
            b, a = np.max(img), np.min(img)
            X[i, 0] = (b-img) / (b-a) if b > a else 0
    
        return X
    
    def load_train_data():
        df = pd.read_csv(train_csv)
        X = _load_rawdata(df, train_images)
        Y = df[_columns].values
        return X, Y
    
    X, y = load_train_data()

    joblib.dump(X,"X.pkl")
    joblib.dump(y,"y.pkl")
else:
    X = joblib.load("X.pkl")
    y = joblib.load("y.pkl")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.8, random_state=1234)
print('学習データの数:', len(X_train))
print('検証データの数:', len(X_test))

"""
fMakeTrain = 0
fDoMode = 0

if fMakeTrain == 1:
    def load_data(file_name, img_dir, img_shape, orientations, pixels_per_cell, cells_per_block):
        classes = ['company_name', 'full_name', 'position_name', 'address', 'phone_number', 'fax', 'mobile', 'email', 'url']
        df = pd.read_csv(file_name)
        n = len(df)
        Y = np.zeros((n, len(classes)))
        print('loading...')
        s = clock()
        for i, row in df.iterrows():
            f, l, t, r, b = row.filename, row.left, row.top, row.right, row.bottom
            print(i,f)
            img = Image.open(os.path.join(img_dir, f)).crop((l,t,r,b)) # 項目領域画像を切り出す
            if img.size[0]<img.size[1]:                                # 縦長の画像に関しては90度回転して横長の画像に統一する
                img = img.transpose(Image.ROTATE_90)
            
            # preprocess
            img_gray = img.convert('L')
            img_gray = np.array(img_gray.resize(img_shape))/255.       # img_shapeに従った大きさにそろえる
    
    
            # feature extraction
            img = np.array(hog(img_gray,orientations = orientations,
                               pixels_per_cell = pixels_per_cell,
                               cells_per_block = cells_per_block))
            if i == 0:
                feature_dim = len(img)
                print('feature dim:', feature_dim)
                X = np.zeros((n, feature_dim))
            
            X[i,:] = np.array([img])
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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.8, random_state=1234)
print('学習データの数:', len(X_train))
print('検証データの数:', len(X_test))
"""

# Network definition
class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__(
        conv1=F.Convolution2D(3, 32, 3, pad=1),
        conv2=F.Convolution2D(32, 32, 3, pad=1),
        l1=F.Linear(2048, 1024),
        l2=F.Linear(1024, 10)
        )
        self.train = True
    
#    def forward(x_data, y_data, train=True):
    def __call__(self, x):#, t):
#        x, t = chainer.Variable(x), chainer.Variable(t)
        x = chainer.Variable(x)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.l1(h)), train=self.train)
        y = self.l2(h)
        return y
#        if self.train:
#            return F.softmax_cross_entropy(y, t)
#        else:
#            return F.accuracy(y, t)

"""
    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
"""

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

batchsize = 100
n_epoch = 20

# Prepare multi-layer perceptron model, defined in net.py
#if args.net == 'simple':
model = L.Classifier(CNN())

#    model = L.Classifier(net.MnistMLP(784, n_units, 10))
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy


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
    perm = np.random.permutation(X_train.shape[0])
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, X_train.shape[0], batchsize):
        x = chainer.Variable(xp.asarray(X_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)

        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph(
                    (model.loss, ), remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / X_train.shape[0], sum_accuracy / X_train.shape[0]))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, X_test.shape[0], batchsize):
        x = chainer.Variable(xp.asarray(X_test[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                             volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / X_test.shape[0], sum_accuracy / X_test.shape[0]))

# Save the model and the optimizer
print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)
