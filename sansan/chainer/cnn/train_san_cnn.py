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
from sklearn.ensemble import RandomForestClassifier#ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from datetime import datetime
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.cross_validation import StratifiedKFold
from xgboost import XGBClassifier
import math
from sklearn.metrics import mean_absolute_error

import os
import time


from time import clock
from PIL import Image

fMakeTrain = 0
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
class MLP(chainer.Chain):
    def __init__(self):
        super(MLP, self).__init__(
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





def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP())
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load data
#    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()


    

