# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:01:35 2016

"""
import cv2
from skimage.feature import hog
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
#from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.cross_validation import StratifiedKFold

import os
import time




#train_path = 'resize_imgs320x240/train'

fMakeTrain = 1
fDoGrid = 0
fDoEvol = 1
width = 320
height = 240
smp_count = 22424

if fMakeTrain == 1:
    X = np.empty((smp_count,width*height),dtype ='uint8')
    y = np.empty((smp_count),dtype ='uint8')
    
    #訓練画像一覧を開きファイル名でソート
    train_csv = pd.read_csv("train.csv")
    train_csv = train_csv.sort("filename")
    train_csv = train_csv.reset_index( drop = True )
    
    old_row_filename = ""
    for i, row in train_csv.iterrows():
    #    print(i,row.filename)
        image_file = "./train_images_1/" + row.filename
    
        if row.filename != old_row_filename:
            src = cv2.imread(image_file,0)
            print(old_row_filename)
            old_row_filename = row.filename
    
        tmp = src[row.top:row.bottom,row.left:row.right]
        
        
#        cv2.imwrite("tmp.png",tmp)
        """
        if tmp.shape[0] > tmp.shape[1]:
            print(row.filename,"縦")
        else:
            print(row.filename,"横")
        """
    
        if i == 80:
            break


    dir_list = ["train_images_1","train_images_2","train_images_3"]#next(os.walk(train_path))[1]
    
    i = 0
#    for d in ['c1']:
    for d in dir_list:
#        t = d.replace('c','') 
#        d = train_path + '/' + d
        files = os.listdir(d)
        for file in files:
            src = cv2.imread(d + '/' + file,0)
            src = cv2.resize(src,(width,height))
            src = src.ravel()
            src = src[:, np.newaxis]
            src = src.T
            X[i] = src
            y[i] = np.uint8(t)
            i = i + 1
   
    joblib.dump(X,"X_"+str(width) + "x" + str(height) + ".pkl",compress=1)
    joblib.dump(y,"y_"+str(width) + "x" + str(height) + ".pkl",compress=1)

else:
    X = joblib.load("X_"+str(width) + "x" + str(height) + ".pkl")
    y = joblib.load("y_"+str(width) + "x" + str(height) + ".pkl")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

if __name__ == '__main__':

    #train
    print('Start train.\n')
    start = time.time()          
    
    if fDoGrid == 1:
        param_grid = {'bootstrap':[False],
                     'criterion': ['entropy'],
                     'max_depth': [19],
                     'max_features': [295],
                     'min_samples_split': [3],
                     'n_estimators': [198],
                     'n_jobs': [-1],
                     "verbose":[1]}
                      
        
        estimator = GridSearchCV(RandomForestClassifier(10),param_grid=param_grid,n_jobs=1,verbose=1)
                                                   
        estimator.fit(X_train, y_train)
    
        elapsed_time = time.time() - start
        print( ("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
    
        print('gridsearchcv best score:',estimator.best_score_)
        train_score = estimator.best_score_

#        joblib.dump(estimator,'estimator.pkl',compress=1)

    if fDoEvol == 1:
        lmax_features = list(range(10,500))
#        lmax_features.append('sqrt')
#        lmax_features.append('log2')
#        lmax_features.append('None')
        param_grid ={'n_estimators': list(range(20,600)),
                     'max_depth': list(range(1,20)),
                     'min_samples_split': list(range(2,8)),
                     'max_features': lmax_features,
#                     'verbose': [1],
                     'criterion': ['entropy','gini'],
                     'bootstrap': [True,False],
#                     'min_samples_leaf': list(range(1,10)),
                    "n_jobs":[-1]}

        estimator = EvolutionaryAlgorithmSearchCV(estimator=RandomForestClassifier(10),
                                   params=param_grid,
                                   scoring="log_loss",
                                   cv=StratifiedKFold(y_train, n_folds=10),
                                   verbose=True,
                                   population_size=80,
                                   gene_mutation_prob=0.10,
                                   tournament_size=3,
                                   generations_number=15,
                                   n_jobs=1)
                                   
        estimator.fit(X_train, y_train)
        elapsed_time = time.time() - start
        print( ("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
         
        print(estimator.best_score_)
        print(estimator.best_params_)
        train_score = estimator.best_score_
        train_params = estimator.best_params_

    pred = estimator.best_estimator_.predict(X_test)
    print('train score',train_score)
    print('test score',accuracy_score(y_test,pred))
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
    
    dir_name = datetime.now().strftime("%Y%m%d_%H%M%S") 
    os.mkdir(dir_name)
    logfilename = dir_name + '/' + 'log.txt'
    with open(logfilename, "w") as file:
        file.write(("elapsed_time:{0}".format(elapsed_time)) + "[sec]\n")
        file.write("width:" + str(width) + " height:" + str(height) + "\n")
        file.write("train sample num:" + str(X_train.shape[0]) + '\n')
        file.write(train_params)
        file.write('\n')
        file.write('train_score:'+ str(train_score) + '\n')
        file.write('test score:' + str(accuracy_score(y_test,pred)) + '\n\n')
        file.write(str(classification_report(y_test, pred))  + '\n')
        file.write(str(confusion_matrix(y_test, pred)))
    
    joblib.dump(estimator,dir_name + '/' + 'estimator.pkl',compress=1)

