#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:43:07 2018

@author: koolok
"""

from multiprocessing import Pool
import pandas as pd
import os
import json

def parse (directory,file) :

    file_ = open(file, "r", encoding="utf-8")
    line = file_.readline()
    
    # Extract image informations
    filename = file.replace(directory,"").replace("json","jpg")
    list_obj = json.loads(line)
    
    #Extract object informations
    objects = list()
    
    for obj in list_obj :
        name = obj.get("label")
        confidence = obj.get("confidence")
        
        xmin = obj.get("topleft").get("x")
        ymin = obj.get("topleft").get("y")
        xmax = obj.get("bottomright").get("x")
        ymax = obj.get("bottomright").get("y")
        
        objects.append((name,confidence,xmin,ymin,xmax,ymax))
        
        
    #Transform objects informations in a Dataframe
    col_names = ['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    obj_list = pd.DataFrame(objects,columns=(col_names))

    return (filename,obj_list)

# Extract Images information
files = list()
for file in os.listdir("out_tiny_full") :
    files.append("out_tiny_full/"+file)
  
img = []
for file in files :
    img.append(parse("out_tiny_full/",file))

# Transfom the result in a Dataframe
col_names = ['filename', 'obj_list']
images = pd.DataFrame(img,columns=(col_names))

images = images.sort_values(by = 'filename')
images.index = range(len(images))

# Save the Dataframe
images.to_pickle("out_tiny_full.pk")




# Extract Images information
files = list()
for file in os.listdir("out_30_epoch_full") :
    files.append("out_30_epoch_full/"+file)
  
img = []
for file in files :
    img.append(parse("out_30_epoch_full/",file))

# Transfom the result in a Dataframe
col_names = ['filename', 'obj_list']
images = pd.DataFrame(img,columns=(col_names))

images = images.sort_values(by = 'filename')
images.index = range(len(images))

# Save the Dataframe
images.to_pickle("out_30_epoch_full.pk")


# Extract Images information
files = list()
for file in os.listdir("out_260_epoch_full") :
    files.append("out_260_epoch_full/"+file)
  
img = []
for file in files :
    img.append(parse("out_260_epoch_full/",file))

# Transfom the result in a Dataframe
col_names = ['filename', 'obj_list']
images = pd.DataFrame(img,columns=(col_names))

images = images.sort_values(by = 'filename')
images.index = range(len(images))

# Save the Dataframe
images.to_pickle("out_260_epoch_full.pk")
