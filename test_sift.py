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
from multiprocessing import Pool
from time import time
import numpy as np
import pandas as pd





#==============================================================================
# Testing for 1 image
#==============================================================================

img = cv2.imread("../VOCdevkit/VOC2007/JPEGImages/000001.jpg")
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)

idf = pd.read_pickle("./images_dataframe.pk")

obj_list = idf['obj_list'][i_img]
X = pd.DataFrame()
y = pd.DataFrame()
for i_kp in range(len(kp)):
    for i_obj in range(len(obj_list)):
        class_name = obj_list['name'][i_obj]
        if ((obj_list['xmin'][i_obj] <= kp[i_kp].pt[0] <= obj_list['xmax'][i_obj])
        and (obj_list['ymin'][i_obj] <= kp[i_kp].pt[1] <= obj_list['ymax'][i_obj])):
            # 1) add descriptor in X
            X = X.append(pd.DataFrame(des[i_kp]).T)
            # 2) add class in y
            y = y.append(pd.DataFrame([class_name]))
            print("hi")
            



good = []

for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
img3 = img

cv2.drawMatchesKnn(img,kp,img2,kp2,good,img3,flags=2)

plt.imshow(img3),plt.show()


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

for i_img in range(len(list_files)):
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
                X = X.append(pd.DataFrame(des[i_kp]).T)
                # 2) add class in y
                y = y.append(pd.DataFrame([obj_list['name'][i_obj]]))
                



# =============================================================================
# Testing    
# =============================================================================
    
bf = cv2.BFMatcher()
descriptors_train = np.array([des,des2])
matches = bf.knnMatch(descriptors_train,k=2)

    
    
    
    