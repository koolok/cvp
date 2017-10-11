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




#==============================================================================
# Testing for 1 image
#==============================================================================

img = cv2.imread("../VOCdevkit/VOC2007/JPEGImages/000001.jpg")
img2 = cv2.imread("../VOCdevkit/VOC2007/JPEGImages/000002.jpg")

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)
kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),None)


bf = cv2.BFMatcher()
descriptors_train = np.array([des,des2])
matches = bf.knnMatch(descriptors_train,k=2)






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

def process(file) :
    print(str(file))
    img = cv2.imread(path+file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)

time_start = time()
pool = Pool()

pool.map_async(process,list_files)


print("time",time()-time_start,"seconds")

    
    
    
    
    
    
    
    
    
    
    
    
    