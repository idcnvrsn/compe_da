# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:52:58 2016
"""

from multiprocessing import Process, Manager
import multiprocessing
import time
import cv2
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import glob
import os

countCore = multiprocessing.cpu_count()
parallel = 1
fLoadEstim = 1

data_final = pd.DataFrame([],columns = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])

test = pd.read_csv("test.csv")

def countDown(estimator,i,files,m,filename_list):

    probs = []
    for file in files:
        src = cv2.imread(file,0)
    #        src = cv2.resize(src,(64,48))
        src = src.ravel()
        src = src[:, np.newaxis]
        src = src.T
    
        prob = estimator.best_estimator_.predict_proba(src)
        print(i,prob)
        probs.append(prob.tolist()[0])

    m.append(probs)
    dfile = [os.path.basename(file) for file in files]
    filename_list.append(dfile)

    print('end',str(i))

if __name__ == '__main__':
    
    start_time = time.time()          

    if fLoadEstim == 1:
        estimator = joblib.load('estimator.pkl')
    
    d = './train_images/*'

#    files = os.listdir(d)
    files = glob.glob(d)
    
    #対象データをコア数分で分ける
    countOf1cpu = int(len(files) / countCore)
    file_lists = []
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
