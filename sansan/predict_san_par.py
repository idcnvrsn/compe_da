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
import glob
import os
from skimage.feature import hog

from time import clock
from PIL import Image

countCore = multiprocessing.cpu_count()
parallel = 1
fLoadEstim = 1

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
        
    print('Done. Took', clock()-s, 'seconds.')
    return X


#def countDown(estimator,i,files,m,filename_list):
def countDown(estimator,i,rows,m):

    probs = []
    for row in rows:
        prob = estimator.best_estimator_.predict_proba(row)
        print(i,prob)
#        probs.append(prob.tolist()[0])
        probs.append(prob)

    m.append(probs)
#    dfile = [os.path.basename(file) for file in files]
#    filename_list.append(dfile)

    print('end',str(i))

if __name__ == '__main__':
    
    start_time = time.time()          

    if fLoadEstim == 1:
        estimator = joblib.load('estimator.pkl')

    img_shape = (216,72)
    orientations = 6
    pixels_per_cell = (12,12)
    cells_per_block = (1, 1)
    X = load_data('test.csv', 'test_images', img_shape, orientations, pixels_per_cell, cells_per_block)
   
    #対象データをコア数分で分ける
    countOf1cpu = int(X.shape[0] / countCore)
    surplus = int(X.shape[0] % countCore)
    
    lists = []
    for i in range(countCore):
        start = countOf1cpu*i
        end = countOf1cpu*i + countOf1cpu
        lists.append(X[start:end])
    lists[countCore-1] = np.vstack([lists[countCore-1],X[end:end+surplus]])
    
    manager = Manager()
    ms = [manager.list() for i in range(countCore)]
    if parallel == 0: 
        countDown(estimator,i[0],lists[0]) 
    else:
        jobs = [Process(target=countDown, args=(estimator,j,lists[j],ms[j],)) for j in range(countCore)]
         
        start_time = time.time()
        for j in jobs:
            j.start()
         
        for j in jobs:
            j.join()
        
        for i in range(countCore):
#            print(ms[i])

            
            probs = ms[i][0]
            df_probs = pd.DataFrame(probs)
            
            data_final = pd.concat([data_final,df_probs])

        data_final.to_csv('submission.csv')#,index=False)

     
    elapsed_time = time.time() - start_time
    print( ("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
    
    
    
    
    
    
    
    
    
    
    
'''
    
    
    #画像を読み込み
    
    lists = []
    for i in range(0,countCore):
        file_lists.append( files[countOf1cpu*i : countOf1cpu*i + countOf1cpu])
    final_list = file_lists[-1]
    final_list = final_list + files[-(len(files) % countCore):]
    file_lists[-1] = final_list
    
    manager = Manager()
    ms = [manager.list() for i in range(countCore)]
    filename_lists = [manager.list() for i in range(countCore)]
    if parallel == 0: 
        countDown(estimator,i[0],file_lists[0]) 
    else:
        jobs = [Process(target=countDown, args=(estimator,j,file_lists[j],ms[j],filename_lists[j],)) for j in range(countCore)]
         
        start_time = time.time()
        for j in jobs:
            j.start()
         
        for j in jobs:
            j.join()
        
        for i in range(countCore):
#            print(ms[i])

            
            df_files = pd.DataFrame(list(filename_lists[i]))
            df_files = df_files.T
            probs = ms[i][0]
            df_probs = pd.DataFrame(probs)
            data = pd.concat([df_files,df_probs],axis=1)
            data.columns = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
            
            data_final = pd.concat([data_final,data])

        data_final.to_csv('submission.csv',index=False)

     
    elapsed_time = time.time() - start_time
    print( ("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
'''