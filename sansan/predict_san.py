# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:53:22 2016
"""

from multiprocessing import Process, Manager
import multiprocessing
import time
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import os
from skimage.feature import hog

from time import clock
from PIL import Image

countCore = multiprocessing.cpu_count()
parallel = 1
fLoadEstim = 1
fMakeTest = 0

data_final = pd.DataFrame([])

test_csv = pd.read_csv("test.csv")

def load_data(file_name, img_dir, img_shape, orientations, pixels_per_cell, cells_per_block):
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
    
#        if i == 100:
#            break
#    X = X[:100]
        
    print('Done. Took', clock()-s, 'seconds.')
    return X


#def countDown(estimator,i,files,m,filename_list):
def countDown(estimator,i,rows,m):

    probs = []
    for j,row in enumerate(rows):
        print("process",j)
#        print(row.shape)
        
        prob = estimator.best_estimator_.predict_proba(row.reshape(1,-1))
#        probs.append(prob.tolist()[0])
#        for p in prob:
#            print(p[0][1])
        
        probs.append([p[0][1] for p in prob])
#        probs.append(prob)

    m.append(probs)

    print('end',str(i))

if __name__ == '__main__':
    
    start_time = time.time()          

    if fLoadEstim == 1:
        estimator = joblib.load('estimator.pkl')

    if fMakeTest == 1:
        img_shape = (216,72)
        orientations = 6
        pixels_per_cell = (12,12)
        cells_per_block = (1, 1)
        X = load_data('test.csv', 'test_images', img_shape, orientations, pixels_per_cell, cells_per_block)
        joblib.dump(X,"test_X.pkl")
    else:
        X = joblib.load("test_X.pkl")

    pred = estimator.best_estimator_.predict(X)
    
    data_final = pd.DataFrame(pred)

    data_final.to_csv('submission.csv',header=False)#,index=False)

     
    elapsed_time = time.time() - start_time
    print( ("elapsed_time:{0}".format(elapsed_time)) + "[sec]")

