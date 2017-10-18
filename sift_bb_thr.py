#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:24:51 2017
building the descriptors within bounding boxes acc. to threshold
@author: jmarnat
"""

#from PIL import Image
import cv2
#import matplotlib.pyplot as plt
from os import listdir
from os import chdir
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


chdir('/home/jmarnat/Documents/CV/cvp-master')

#==============================================================================
# Training on the whole dataset
#==============================================================================

"""List of all image files""" 
path = "../VOCdevkit/VOC2007/JPEGImages/"
list_files = listdir(path)
list_files.sort()

# setting the contrast threshold to 0.15 allows us to restrict the number
# of keypoints to arrount 100 per object within each picture.
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.2)
idf = pd.read_pickle("./images_dataframe.pk")

# creating the function to process
def buildClasses(i_img):
    X_tmp = pd.DataFrame()
    y_tmp = pd.DataFrame()
    print(i_img,"/",len(list_files))
      
    obj_list = idf['obj_list'][i_img]
    img = cv2.imread(path+list_files[i_img])
    for i_obj in range(len(obj_list)):
        xmin = obj_list['xmin'][i_obj]
        xmax = obj_list['xmax'][i_obj]
        ymin = obj_list['ymin'][i_obj]
        ymax = obj_list['ymax'][i_obj]
        obj_name = [obj_list['name'][i_obj]]
        
        img_obj = img[ymin:ymax,xmin:xmax]
        
        kp, des = sift.detectAndCompute(cv2.cvtColor(img_obj,cv2.COLOR_BGR2GRAY),None)  
        
        # 1) add descriptor in X
        X_tmp = X_tmp.append(pd.DataFrame(des))
        # 2) add class in y
        y_tmp = y_tmp.append(pd.DataFrame(obj_name*len(kp)))
    return [X_tmp,y_tmp]

pool = Pool()
list_Xy = pool.map_async(buildClasses,range(len(list_files))).get()
#list_Xy = pool.map_async(buildClasses,range(100)).get()



# dataframe of the keypoints's descriptors
X = pd.DataFrame()

# classes associated to the keypoints
y = pd.DataFrame()

# testing on 1000/45xx images
# TODO go through all then
#a=list_Xy[1:100]
a = list_Xy
for i_df in range(len(a)):
    X = X.append(pd.DataFrame(a[i_df][0]))
    y = y.append(pd.DataFrame(a[i_df][1]))

print(len(X)/100)


X.to_csv("X_sifts-0.20.csv")
y.to_csv("y_sifts-0.20.csv")


#for i_img in range(len(list_files)):

#
## =============================================================================
## Testing    
## =============================================================================
#    
#bf = cv2.BFMatcher()
#
#
#X_train = pd.read_csv('X_new-1000-imgs.csv')
#X_train = X_train.drop('Unnamed: 0',axis=1)
#X_train = np.array(X_train)
#
#y_train = pd.read_csv('y_new-1000-imgs.csv')
#y_train = y_train.drop('Unnamed: 0',axis=1)
#y_train = y_train.values
#y_train = np.asarray(y_train)
##y_train = np.ones(len(y_train))
#y_train = y_train.astype(str)
#
#
#            
#classes = ['person','bird','cat','cow','dog',
#           'horse','sheep','aeroplane','bicycle','boat',
#           'bus','car','motorbike','train','bottle',
#           'chair','diningtable','pottedplant','sofa','tvmonitor']
#
#classes = pd.DataFrame(classes)
#classes = classes.rename(columns={0:'class'})
#
##replacing text classes by number
#for y_i in range(len(y_train)):
#    y_train[y_i] = np.where(classes==y_train[y_i])[0][0]
#y_train = y_train.astype(int)
#
#
#y_train_int = pd.read_csv("y_train_int.csv")
#y_train_int.drop(y_train_int.columns[[0]],axis=1)
#y_train = y_train_int.values
#
#img = cv2.imread(path+"000028.jpg")
#sift = cv2.xfeatures2d.SIFT_create()
#img_kp, img_des = sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)    
#
#
## TESTING WITH SKLEARN
#from sklearn.neighbors import KNeighborsClassifier
#
## train // test
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(
#        X_train, y_train, test_size=0.33, random_state=42)
#
#knn = KNeighborsClassifier(n_neighbors=20)
#knn.fit(X_train,y_train)
#
#predict = knn.predict(X_test[[range(100)]])
#predict_prob = knn.predict_proba(X_test[[range(100)]])
#
#print(predict[[range(10)]],"\n---\n",y_test[[range(10)]])
#predict
#
#print(accuracy_score(y_test[range(100),1].T,predict[:,1].T))
#
#
#











    
    