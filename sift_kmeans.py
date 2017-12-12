    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:07:45 2017

@author: jmarnat
"""

#from PIL import Image
#import cv2
import matplotlib.pyplot as plt
from os import listdir
#from os import chdir
#from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#import kmeans
import cv2


# =============================================================================
# loading the sifts descriptors
# =============================================================================

X = pd.read_csv('X_sifts-0.20.csv')
X = pd.read_csv('X_surf-5000.csv')

X = X.drop('Unnamed: 0',axis=1)
X = np.array(X)

y = pd.read_csv('y_sifts-0.20.csv')
y = pd.read_csv('y_surf-5000.csv')
y = y.drop('Unnamed: 0',axis=1)
y = y.values
#y = np.asarray(y)
y = y.astype(str)

classes = ['person','bird','cat','cow','dog',
           'horse','sheep','aeroplane','bicycle','boat',
           'bus','car','motorbike','train','bottle',
           'chair','diningtable','pottedplant','sofa','tvmonitor']

classes = pd.DataFrame(classes)
classes = classes.rename(columns={0:'class'})

#replacing text classes by number
for y_i in range(len(y)):
    y[y_i] = np.where(classes==y[y_i])[0][0]
y = y.astype(int)




print("DATA LOADED")

# =============================================================================
# clustering
# =============================================================================

# takes 1 minute for len(X) = 158650



X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)


    
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)


km = KMeans(n_clusters=200,max_iter=10,n_jobs=-1).fit(X)
clusters = km.cluster_centers_

pd.DataFrame(clusters).to_csv('clusters-surf.csv')


# =============================================================================
# inputing image and classifying it
# =============================================================================

"""List of all image files""" 
path = "../VOCdevkit/VOC2007/JPEGImages/"
#list_files = listdir(path)
#list_files.sort()

#sift = cv2.xfeatures2d.SIFT_create()
img = cv2.imread(path+"000007.jpg")
plt.imshow(img)
#kp, des = sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)
surf = cv2.xfeatures2d.SURF_create(5000)
kp, des = surf.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)



def predict_tab(km, des):
    res = km.predict(des)
    #res = km.fit_predict(des)
    hist = np.histogram(res,bins=20)[0]

    classes2 = np.asarray(classes)
    df = pd.DataFrame()
    for i in range(len(hist)):
        nrow = pd.DataFrame([classes2[i][0],hist[i]]).T
        df = df.append(nrow)
    
    df = df.sort_values(by=1,ascending=False)
    return(df)

def predict_hist(km, des):
    res = km.predict(des)
    #res = km.fit_predict(des)
    hist = np.histogram(res,bins=20)[1]
    return(pd.DataFrame(hist))


# =============================================================================
# classifying all the test images
# =============================================================================
surf = cv2.xfeatures2d.SURF_create(5000)
path = "../VOCdevkit_test/VOC2007/JPEGImages/"
list_files = listdir(path)
list_files.sort()
n = len(list_files)
n = 10
idf = pd.read_pickle("./images_dataframe_test.pk")
predictions = pd.DataFrame()
realclasses = pd.DataFrame()
for i_img in range(n):
    img = cv2.imread(path+list_files[i_img])
    plt.imshow(img)
    plt.show()
    kp, des = surf.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)
    pred = predict_hist(km, des)
    real_class = idf[idf.index == 1]
    predictions = predictions.append((pd.DataFrame([list_files[i_img]]).append(pred)).T)
#    obj_list = idf['obj_list'][idf.index == 1]
#    for i_obj in range(len(obj_list)):
        
predictions.index = range(len(predictions.index))









