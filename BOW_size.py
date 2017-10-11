#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
from time import time
import pandas as pd
from os import listdir
import pickle as pk
from multiprocessing import Pool

"""Parameters and constants"""
size_image = 50
gap = 4

"""Constants"""
nb_bin = pow(gap,3)
hist_save = [0]*nb_bin
thresh = 256/gap

"""Functions"""
def threshold(component) :
    value = 0
    while component > thresh*(value+1) :
        value += 1
    return value

def process(file) :
    picture = Image.open(path+file)
    width,height = picture.size
    
    nb_width = width//size_image
    nb_height = height//size_image
    square_size = size_image*size_image

    list_hist = [file]
    
    """Histograms of the percentage of pixels in each bin for all squares"""
    for w in range(nb_width) :
        for h in range(nb_height) :
            hist = hist_save.copy()
            
            for W in range(size_image) :
               for H in range(size_image) :
                   r,g,b = picture.getpixel((size_image*w+W,size_image*h+H))
                   hist[threshold(r)+gap*threshold(g)+gap*gap*threshold(b)] += 1
            
            hist = [nb / square_size for nb in hist]
            list_hist.append(hist)

    return list_hist

"""List of all image files""" 
path = "../VOCdevkit/VOC2007/JPEGImages/"
list_files = listdir(path)

time_start = time()

pool = Pool()

list_image = pool.map_async(process,list_files).get()


dataframe = pd.DataFrame(list_image)
print(dataframe)

print("time ",time()-time_start," seconds")
    
with open('BOW_size_'+str(size_image)+'x'+str(size_image)+'_dataframe', 'wb') as file:
    pickler = pk.Pickler(file)
    pickler.dump(dataframe)



