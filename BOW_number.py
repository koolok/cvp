#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
from time import time
import pandas as pd
from os import listdir
import pickle as pk
from multiprocessing import Pool

"""Parameters"""
nb_image = 16
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
    
    new_width = width//nb_image
    new_height = height//nb_image
    rectangle_size = new_width*new_height

    list_hist = [file]
    
    """Histograms of the percentage of pixels in each bin for all rectangles"""
    for w in range(nb_image) :
        for h in range(nb_image) :
            hist = hist_save.copy()
            
            for W in range(new_width) :
               for H in range(new_height) :
                   r,g,b = picture.getpixel((new_width*w+W,new_height*h+H))
                   hist[threshold(r)+gap*threshold(g)+gap*gap*threshold(b)] += 1
            
            hist = [nb / rectangle_size for nb in hist]
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

print("time",time()-time_start,"seconds")
    
with open('BOW_number_'+str(nb_image*nb_image)+'_dataframe', 'wb') as file:
    pickler = pk.Pickler(file)
    pickler.dump(dataframe)



