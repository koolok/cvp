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


def buildClasses(i_img, hess):
    X_tmp = pd.DataFrame()
    y_tmp = pd.DataFrame()
    obj_list = idf['obj_list'][i_img]
    img = cv2.imread(path+list_files[i_img])
    for i_obj in range(len(obj_list)):
        xmin = obj_list['xmin'][i_obj]
        xmax = obj_list['xmax'][i_obj]
        ymin = obj_list['ymin'][i_obj]
        ymax = obj_list['ymax'][i_obj]
        obj_name = [obj_list['name'][i_obj]]
        img_obj = img[ymin:ymax,xmin:xmax]
        
        surf = cv2.xfeatures2d.SURF_create(hess)
        kp, des = surf.detectAndCompute(cv2.cvtColor(img_obj,cv2.COLOR_BGR2GRAY),None)
        
        if len(kp) > 0:
            X_tmp = X_tmp.append(pd.DataFrame(des))
            y_tmp = y_tmp.append(pd.DataFrame(obj_name*len(des)))
    return X_tmp,y_tmp


"""List of all image files""" 
path = "../VOCdevkit/VOC2007/JPEGImages/"
list_files = listdir(path)
list_files.sort()
idf = pd.read_pickle("./images_dataframe.pk")

i_img = 0
i_obj = 0

X = pd.DataFrame()
y = pd.DataFrame()
hess = 5000
n = len(list_files)
#n = 100
for i_img in range(n):
    print((int)(100*i_img/n),'/100')
    X_tmp, y_tmp = buildClasses(i_img, hess)
    X = X.append(X_tmp)
    y = y.append(y_tmp)

# =============================================================================
# test for the best n
# =============================================================================
if False:
    d = pd.DataFrame()
    for hess in range(100,10000,100):
        X = pd.DataFrame()
        y = pd.DataFrame()
        print('hess: ',hess)
        for i_img in range(n):
            X_tmp, y_tmp = buildClasses(i_img,hess)
            X = X.append(X_tmp)
            y = y.append(y_tmp)
            
        d = d.append(pd.DataFrame([hess,len(X)]).T)
    d = d.rename(columns={0:'n',1:'d'})
    plt.scatter(d['n'],d['d'])
    plt.title('#SURFs vs. hessian value')
    plt.ylabel('#SURFs')
    plt.xlabel('hessian value')
    plt.show()
    
    
    


print("DONE POOLING")

# testing on 1000/45xx images

X.index = range(len(X.index))
y.index = range(len(y.index))


print(len(X))

print("DONE APPENDING")

X.to_csv("X_surf-5000.csv")
y.to_csv("y_surf-5000.csv")

print('ALL DONE')





    
    