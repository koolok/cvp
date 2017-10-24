#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pk
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import random as rd
from time import time

"""Parameters"""
nb_clusters = 64

"""Functions"""
def max_dico(dico) :
    if len(dico) == 1 :
        for k in dico :
            return k
    
    maxi = None
    label = None
    
    for k in dico :
        if maxi is None or dico[k] > maxi:
            maxi = dico[k]
            label = k
    return label

"""Recovery of the dataframe"""
#file_name = input("Name of the file to recover : """)
file_name = "BOW_number_16_dataframe"

if len(file_name) == 23 :
    size = int(file_name[11:13])
else :
    size = int(file_name[11:14])
    
with open(file_name, 'rb') as file:
    unpickler = pk.Unpickler(file)
    dataframe = unpickler.load()
    
dataframe_file = dataframe.get('filename')
dataframe = dataframe.get('histograms')

"""Creation of the list of the file"""
list_files = []
for file in dataframe_file :
    list_files += [file]*size

list_files = np.array(list_files)
#print(list_file)

"""Recovery of the labels and creation of the list"""
file_info = "images_dataframe.pk"
with open(file_info, 'rb') as file:
    unpickler = pk.Unpickler(file)
    data_info = unpickler.load()

list_labels = []
for i in range(len(data_info)) :
    dico = dict()
    for j,row in enumerate(data_info.obj_list[i].values) :
        area = (row[6]-row[4])*(row[7]-row[5])
        try :
            dico[row[0]]+= area 
        except :
            dico[row[0]]= area
    list_labels += [max_dico(dico)]*size
    
list_labels = np.array(list_labels)
#print(list_labels)

"""Processing on the dataframe"""
"""Recovery of the histograms and creation of the list"""
list_hist = []

for i in range(len(dataframe)) :
    print(i)
    list_hist += dataframe[i]

list_hist = np.array(list_hist)
#print(list_hist)

"""K means"""
time_start = time()

kmeans = KMeans(n_clusters=nb_clusters, random_state=rd.seed()).fit(list_hist)

time = time()-time_start
print("time ",time," seconds")

#print(kmeans.labels_)
#print(kmeans.cluster_centers_)

#print(list_files[kmeans.labels_ == 0])
#print(list_labels[kmeans.labels_ == 0])

list_index = np.array(range(len(list_files)))
list_labels_clusters = []
list_files_clusters = []
list_index_clusters = []
for i in range (nb_clusters) :
    list_index_clusters.append([list_index[kmeans.labels_ == i]])
    list_labels_clusters.append([list_labels[kmeans.labels_ == i]])
    list_files_clusters.append([list_files[kmeans.labels_ == i]])

#voir colornames