#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:24:51 2017

@author: jmarnat
"""

#from PIL import Image
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os import chdir
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


chdir('/home/jmarnat/Documents/CV/cvp-master')


#==============================================================================
# Testing for 1 image
#==============================================================================

img = cv2.imread("../VOCdevkit/VOC2007/JPEGImages/000001.jpg")
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)

idf = pd.read_pickle("./images_dataframe.pk")


#==============================================================================
# Training on the whole dataset
#==============================================================================

"""List of all image files""" 
path = "../VOCdevkit/VOC2007/JPEGImages/"
list_files = listdir(path)
list_files.sort()

sift = cv2.xfeatures2d.SIFT_create()
idf = pd.read_pickle("./images_dataframe.pk")

# dataframe of the keypoints's descriptors
X = pd.DataFrame()

# classes associated to the keypoints
y = pd.DataFrame()

# creating the function to process
def buildClasses(i_img):
    X_tmp = pd.DataFrame()
    y_tmp = pd.DataFrame()
    print(i_img,"/",len(list_files))
    img = cv2.imread(path+list_files[i_img])
    kp, des = sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)    
    obj_list = idf['obj_list'][i_img]
    for i_kp in range(len(kp)):
        for i_obj in range(len(obj_list)):
            # check if the keypoint is in the bounding box of the object
            if ((obj_list['xmin'][i_obj] <= kp[i_kp].pt[0] <= obj_list['xmax'][i_obj])
            and (obj_list['ymin'][i_obj] <= kp[i_kp].pt[1] <= obj_list['ymax'][i_obj])):
                # 1) add descriptor in X
                X_tmp = X_tmp.append(pd.DataFrame(des[i_kp]).T)
                # 2) add class in y
                y_tmp = y_tmp.append(pd.DataFrame([obj_list['name'][i_obj]]))
    return [X_tmp,y_tmp]

pool = Pool()
list_Xy = pool.map_async(buildClasses,range(len(list_files))).get()

# appending the whole xy list
X_new = pd.DataFrame()
y_new = pd.DataFrame()

# testing on 1000/45xx images
# TODO go through all then
a=list_Xy[1:1000]
for i_df in range(len(a)):
    X_new = X_new.append(pd.DataFrame(a[i_df][0]))
    y_new = y_new.append(pd.DataFrame(a[i_df][1]))


X_new.to_csv("X_new-1000-imgs.csv")
y_new.to_csv("y_new-1000-imgs.csv")


#for i_img in range(len(list_files)):


# =============================================================================
# Testing    
# =============================================================================
    
bf = cv2.BFMatcher()


X_train = pd.read_csv('X_new-1000-imgs.csv')
X_train = X_train.drop('Unnamed: 0',axis=1)
X_train = np.array(X_train)

y_train = pd.read_csv('y_new-1000-imgs.csv')
y_train = y_train.drop('Unnamed: 0',axis=1)
y_train = y_train.values
y_train = np.asarray(y_train)
#y_train = np.ones(len(y_train))
y_train = y_train.astype(str)


            
classes = ['person','bird','cat','cow','dog',
           'horse','sheep','aeroplane','bicycle','boat',
           'bus','car','motorbike','train','bottle',
           'chair','diningtable','pottedplant','sofa','tvmonitor']

classes = pd.DataFrame(classes)
classes = classes.rename(columns={0:'class'})

#replacing text classes by number
for y_i in range(len(y_train)):
    y_train[y_i] = np.where(classes==y_train[y_i])[0][0]
y_train = y_train.astype(int)


y_train_int = pd.read_csv("y_train_int.csv")
y_train_int.drop(y_train_int.columns[[0]],axis=1)
y_train = y_train_int.values

img = cv2.imread(path+"000028.jpg")
sift = cv2.xfeatures2d.SIFT_create()
img_kp, img_des = sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)    


# TESTING WITH SKLEARN
from sklearn.neighbors import KNeighborsClassifier

# train // test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)

predict = knn.predict(X_test[[range(100)]])
predict_prob = knn.predict_proba(X_test[[range(100)]])

print(predict[[range(10)]],"\n---\n",y_test[[range(10)]])
predict

print(accuracy_score(y_test[range(100),1].T,predict[:,1].T))














    
    