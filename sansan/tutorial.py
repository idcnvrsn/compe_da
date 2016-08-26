# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.externals import joblib

df = pd.read_csv('train.csv')
df = df.sort("filename")
df = df.reset_index( drop = True )
df.head()

import os
import numpy as np
from time import clock
from PIL import Image

img = Image.open(os.path.join('train_images', df.ix[0].filename))                      # 2842.pngの読み込み
img_cropped = img.crop((df.ix[0].left, df.ix[0].top, df.ix[0].right, df.ix[0].bottom)) # cropメソッドにより項目領域を切り取る
img_cropped

img_2 = Image.open(os.path.join('train_images', df.ix[2].filename))
img_2_cropped = img_2.crop((df.ix[2].left, df.ix[2].top, df.ix[2].right, df.ix[2].bottom))
img_2_cropped

img_cropped = img_cropped.convert('L') # convertメソッドによりグレースケール化
print(img_cropped.size)

img_resized = img_cropped.resize((216, 72))  # resizeメソッドにより画像の大きさを変える
img_resized

img_array = np.array(img_resized)
print(img_array.shape)

from skimage.feature import hog

img_data = np.array(hog(img_array,orientations = 6,
                        pixels_per_cell = (12, 12),
                        cells_per_block = (1, 1)))     
print(img_data.shape)

def load_data(file_name, img_dir, img_shape, orientations, pixels_per_cell, cells_per_block):
    classes = ['company_name', 'full_name', 'position_name', 'address', 'phone_number', 'fax', 'mobile', 'email', 'url']
    df = pd.read_csv(file_name)
    n = len(df)
    Y = np.zeros((n, len(classes)))
    print('loading...')
    s = clock()
    for i, row in df.iterrows():
        f, l, t, r, b = row.filename, row.left, row.top, row.right, row.bottom
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
    
from sklearn.cross_validation import train_test_split

img_shape = (216,72)
orientations = 6
pixels_per_cell = (12,12)
cells_per_block = (1, 1)
X, Y = load_data('train.csv', 'train_images', img_shape, orientations, pixels_per_cell, cells_per_block)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1234)
print('学習データの数:', len(x_train))
print('検証データの数:', len(x_test))
joblib.dump(X,"X.pkl")
joblib.dump(Y,"Y.pkl")

from sklearn.linear_model import LogisticRegression

class MultiLabelLogistic():
    def __init__(self, n_out):
        self.n_out = n_out
        model_list = []
        for l in range(self.n_out):
            model_list.append(LogisticRegression())
        self.models = model_list
        
    def fit(self, X, Y):
        i = 0
        start_overall = clock()
        for model in self.models:
            start = clock()
            print('training model No.%s...'%(i+1))
            model.fit(X, Y[:,i])
            print('Done. Took', clock()-start, 'seconds.')
            i += 1
        print('Done. Took', clock()-start_overall, 'seconds.')
    
    def predict(self, X):
        i = 0
        predictions = np.zeros((len(X), self.n_out))
        start = clock()
        print('predicting...')
        for model in self.models:
            predictions[:,i] = model.predict(X)
            print(str(i+1),'/',str(self.n_out))
            i += 1
        print('Done. Took', clock()-start, 'seconds.')
        
        return predictions

model = MultiLabelLogistic(n_out = 9)  # 今回は9項目あるため, クラス数は9個に設定
model.fit(x_train, y_train)

predictions = model.predict(x_test)

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))
    
print('MAE:', mae(y_test, predictions))

single = np.where(y_test.sum(axis=1)==1)
print('num of samples (single label):', len(single[0]))
print('MAE:', mae(y_test[single], predictions[single]))

double = np.where(y_test.sum(axis=1)==2)
print('num of samples (double label):', len(double[0]))
print('MAE:', mae(y_test[double], predictions[double]))

triple = np.where(y_test.sum(axis=1)==3)
print('num of samples (triple label):', len(triple[0]))
print('MAE:', mae(y_test[triple], predictions[triple]))

quadruple = np.where(y_test.sum(axis=1)==4)
print('num of samples (quadruple label):', len(quadruple[0]))
print('MAE:', mae(y_test[quadruple], predictions[quadruple]))

print('prediction:\n', predictions[quadruple].astype(np.int))
print('ground truth:\n', y_test[quadruple].astype(np.int))


