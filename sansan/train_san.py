# -*- coding: utf-8 -*-
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
    df = pd.read_csv('train.csv')
    df = df.sort_values(by=["filename"])
    df = df.reset_index( drop = True )
    df.head()

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
    X, y = load_data('train.csv', 'train_images', img_shape, orientations, pixels_per_cell, cells_per_block)
#    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1234)
    joblib.dump(X,"X.pkl")
    joblib.dump(y,"y.pkl")
else:
    X = joblib.load("X.pkl")
    y = joblib.load("y.pkl")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
print('学習データの数:', len(X_train))
print('検証データの数:', len(X_test))

if __name__ == '__main__':

    #train

    print('Start train.\n')
    start = time.time()          
  
#    k = ( 1 + int(math.log(len(X_train))/math.log(2)) ) #* 4    
    if fDoMode == 0:
        param_grid = {#'bootstrap':[False],
                     #'criterion': ['entropy'],
                     #'max_depth': [19],
                     #'max_features': [100],
                     #'min_samples_split': [3],
                     'n_estimators': [150],
                     'n_jobs': [-1],
                     "verbose":[1]}
                      
        
        estimator = GridSearchCV(RandomForestClassifier(10),param_grid=param_grid,cv=10,n_jobs=1,verbose=1)
        #cv = StratifiedKFold(y_train, n_folds=3)
        '''                                                   

        param_grid = {"n_estimators":[100],
                      'objective':['multi:softprob'],
    #                  "max_depth": [3, None],
#                      "max_features": ['sqrt', 'None'],
    #                  "min_samples_split": [1, 3, 10],
    #                  "min_samples_leaf": [1, 3, 10],
    #                  "bootstrap": [True, False],
#                      "n_jobs": [-1],
#                      "verbose":[1]
                      "nthread":[-1]
                      }
        estimator = GridSearchCV(XGBClassifier(10),param_grid=param_grid,cv = 10,n_jobs=1,verbose=1)
        '''


        estimator.fit(X_train, y_train)
    
        elapsed_time = time.time() - start
        print( ("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
    
        print('gridsearchcv best score:',estimator.best_score_)
        train_score = estimator.best_score_

#        joblib.dump(estimator,'estimator.pkl',compress=1)

    if fDoMode == 1:
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
#    print(confusion_matrix(y_test, pred))
    
    def mae(y, yhat):
        return np.mean(np.abs(y - yhat))
    
    print('MAE:', mae(y_test, pred))
    
    dir_name = datetime.now().strftime("%Y%m%d_%H%M%S") 
    os.mkdir(dir_name)
    logfilename = dir_name + '/' + 'log.txt'
    with open(logfilename, "w") as file:
        file.write(("elapsed_time:{0}".format(elapsed_time)) + "[sec]\n")
        file.write("train sample num:" + str(X_train.shape[0]) + '\n')
        file.write(str(train_params))
        file.write('\n')
        file.write('train_score:'+ str(train_score) + '\n')
        file.write('test score:' + str(accuracy_score(y_test,pred)) + '\n')
        file.write('MAE:' + str(mae(y_test, pred)) + '\n\n')
        file.write(str(classification_report(y_test, pred))  + '\n')
#        file.write(str(confusion_matrix(y_test, pred)))
    
    joblib.dump(estimator,dir_name + '/' + 'estimator.pkl',compress=1)


