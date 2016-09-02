# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:19:50 2016
"""

from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from time import clock
from PIL import Image
from PIL import ImageOps
import os
from skimage.transform import resize

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import net

## データの読み込み
data_final = pd.DataFrame([])

fMakeTest = 0
if fMakeTest == 1:
    
    def load_data(file_name, img_dir, img_shape, orientations, pixels_per_cell, cells_per_block):
        classes = ['company_name', 'full_name', 'position_name', 'address', 'phone_number', 'fax', 'mobile', 'email', 'url']
        df = pd.read_csv(file_name)
        n = len(df)
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
        
        print('Done. Took', clock()-s, 'seconds.')
        return X
    
    img_shape = (216,72)
    orientations = 6
    pixels_per_cell = (12,12)
    cells_per_block = (1, 1)
    X = load_data('../../test.csv', '../../test_images', img_shape, orientations, pixels_per_cell, cells_per_block)
#    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1234)
    joblib.dump(X,"test_X.pkl")
else:
    X = joblib.load("test_X.pkl")

n_units = 1000

xp = np
x = chainer.Variable(xp.asarray(X))

#９個の分類器をテストデータの各行に適用
pred = np.zeros((X.shape[0],9),dtype=X.dtype)
for i_cls in range(9):
    print(i_cls)
    #予測
    model = L.Classifier(net.MnistMLP(15000, n_units, 2))
    
    serializers.load_npz('mlp' + str(i_cls) + '.model', model)
    
    pred_raw = model.to_cpu().predictor(x)
#        np.max(pred.data,axis=1)
    
    pred_bin = np.argmax(pred_raw.data,axis=1)
    
    pred[:,i_cls] = pred_bin

data_final = pd.DataFrame(pred)

data_final.to_csv('submission.csv',header=False)#,index=False)


