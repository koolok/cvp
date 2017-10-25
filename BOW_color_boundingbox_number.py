#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from time import time
import pandas as pd
from os import listdir
import pickle as pk
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

"""Parameters"""
nb_bin = 16

"""Constants"""
color = ('b','g','r')

"""Recuperation of the informations"""
idf = pd.read_pickle("./images_dataframe.pk")

"""List of all image files""" 
path = "../VOCdevkit/VOC2007/JPEGImages/"
list_files = listdir(path)

"""Function"""
def process(i_img):
    print(i_img,"/",len(list_files))
      
    obj_list = idf['obj_list'][i_img]
    img = cv2.imread(path+list_files[i_img])
    list_hist = []
    for i_obj in range(len(obj_list)):
        xmin = obj_list['xmin'][i_obj]
        xmax = obj_list['xmax'][i_obj]
        ymin = obj_list['ymin'][i_obj]
        ymax = obj_list['ymax'][i_obj]
        obj_name = obj_list['name'][i_obj]
        
        """create a mask"""
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[ymin:ymax,xmin:xmax] = 255
        
        """build the histogram"""
        hist = []
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],mask,[256],[0,nb_bin])
            hist += (histr[range(0,256,int(256/nb_bin))]/histr.sum()).flatten().tolist()
#            print(histr,histr.sum())
#            plt.plot(histr,color = col)
#            plt.xlim([0,nb_bin])
#            plt.title(str(i_img)+' '+str(obj_name)+' '+str(col))
#            plt.show()
        ret = [obj_name]
        ret += hist
        list_hist.append(ret)
    return(list_hist)

time_start = time()

pool = Pool()

list_image = pool.map_async(process,range(len(list_files))).get()
list_image = [item for liste in list_image for item in liste]

pool.close()

dataframe = pd.DataFrame(list_image)

print("time ",time()-time_start," seconds")

print(dataframe)

with open('BOW_color_boundingbox_'+str(nb_bin)+'_dataframe', 'wb') as file:
    pickler = pk.Pickler(file)
    pickler.dump(dataframe)



