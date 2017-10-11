#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:37:58 2017

@author: koolok
"""

from multiprocessing import Pool
from lxml import etree
import pandas as pd
import os

# Take a xml file and return tuple (name,pose,truncated,difficult,xmin,ymin,xmax,ymax)
def parse (file) :
        
    # Create xml tree from the file
    tree = etree.parse(file)
            
    # Extract image informations
    filename = tree.xpath('/annotation/filename')[0].text
    width = int(tree.xpath('/annotation/size/width')[0].text)
    height = int(tree.xpath('/annotation/size/height')[0].text)
    depth = int(tree.xpath('/annotation/size/depth')[0].text)
    
    #Extract object informations
    objects = list()
    
    for obj in tree.xpath('/annotation/object') :
        name = obj.xpath('./name')[0].text
        pose = obj.xpath('./pose')[0].text
        truncated = int(obj.xpath('./truncated')[0].text)
        difficult = int(obj.xpath('./difficult')[0].text)
        
        xmin = int(obj.xpath('./bndbox/xmin')[0].text)
        ymin = int(obj.xpath('./bndbox/ymin')[0].text)
        xmax = int(obj.xpath('./bndbox/xmax')[0].text)
        ymax = int(obj.xpath('./bndbox/ymax')[0].text)
        
        objects.append((name,pose,truncated,difficult,xmin,ymin,xmax,ymax))
        
        
    #Transform objects informations in a Dataframe
    col_names = ['name', 'pose', 'truncated', 'difficult', 'xmin', 'ymin', 'xmax', 'ymax']
    obj_list = pd.DataFrame(objects,columns=(col_names))

    return (filename,width,height,depth,obj_list)



# Create a pool of process
pool = Pool()

# Extract Images information
files = list()
for file in os.listdir("../VOCdevkit/VOC2007/Annotations") :
    files.append("../VOCdevkit/VOC2007/Annotations/"+file)
    
img = pool.map_async(parse, files).get()
    
# Transfom the result in a Dataframe
col_names = ['filename', 'width', 'height', 'depth', 'obj_list']
images = pd.DataFrame(img,columns=(col_names))

images = images.sort_values(by = 'filename')
images.index = range(len(images))

# Save the Dataframe
images.to_pickle("images_dataframe.pk")
