#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pk
from sklearn.cluster import KMeans
import numpy as np
from time import time

"""Recovery of the labels"""
file_info = "images_dataframe.pk"
with open(file_info, 'rb') as file:
    unpickler = pk.Unpickler(file)
    data_info = unpickler.load()

data_info = data_info.sort_values(by = 'filename')
data_info.index = range(len(data_info))
print(data_info.filename)
    
#"""Recovery of the dataframe"""
#"""Attention modification possible de la double boucle for mettre la valeur du 
#fichier +1 pour j"""
##file_name = input("Name of the file to recover : """)
#file_name = "BOW_number_16_dataframe"
#
#with open(file_name, 'rb') as file:
#    unpickler = pk.Unpickler(file)
#    dataframe = unpickler.load()
#
#"""Processing on the dataframe"""
#dataframe_file = dataframe.truncate(after = 0,axis =1)
#dataframe = dataframe.truncate(1,axis =1)
#
#list_hist = []
#list_file = []
#
#for i in range(len(dataframe)) :
#    for j in range(1,17) :
#        list_hist.append(dataframe[j][i])
#
#time_start = time()
#
#X = np.array(list_hist)
#kmeans = KMeans(n_clusters=64, random_state=0).fit(X)
#
#print("time ",time()-time_start," seconds")
#print(kmeans.labels_)
#print(kmeans.cluster_centers_)        
