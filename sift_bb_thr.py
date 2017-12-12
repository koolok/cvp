#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:24:51 2017
building the descriptors within bounding boxes acc. to threshold
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
#from sklearn.metrics import accuracy_score
import random as rand


chdir('/home/jmarnat/Documents/CV-project/cvp')

#==============================================================================
# Training on the whole dataset
#==============================================================================

"""List of all image files""" 
path = "../VOCdevkit/VOC2007/JPEGImages/"
list_files = listdir(path)
list_files.sort()

# setting the contrast threshold to 0.15 allows us to restrict the number
# of keypoints to arrount 100 per object within each picture.
sift = cv2.xfeatures2d.SIFT_create()
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
        
        # todo : keep only 100 random points:
        nb_points = np.min([len(kp),100])
#        print('nb_points = ',nb_points)
        random_list = rand.sample(range(len(kp)),nb_points)
        
#        kp_tmp = pd.DataFrame()
        des_tmp = pd.DataFrame()
        
        for idx in (random_list):
            # kp_tmp = kp_tmp.append(pd.DataFrame(kp[idx]))
            des_tmp = des_tmp.append(pd.DataFrame(des[idx]).T)
        

        # 1) add descriptor in X
        X_tmp = X_tmp.append(des_tmp)
        # 2) add class in y
        y_tmp = y_tmp.append(pd.DataFrame(obj_name*nb_points))
    return [X_tmp,y_tmp]

pool = Pool()

list_Xy = pool.map_async(buildClasses,range(len(list_files))).get()
#list_Xy = pool.map_async(buildClasses,range(100)).get()

print("DONE POOLING")

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

X.index = range(len(X.index))
y.index = range(len(y.index))


print(len(X))

print("DONE APPENDING")

X.to_csv("X_sifts-max100.csv")
y.to_csv("y_sifts-max100.csv")

print('ALL DONE')





    
    