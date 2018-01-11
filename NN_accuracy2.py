#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 20:36:03 2018

@author: koolok
"""

import pickle

file = open("out_260_epoch_full.pk", 'rb')
result = pickle.load(file)

file = open("images_dataframe_test.pk", 'rb')
original = pickle.load(file)

the_classes = ['person','bird','cat','cow','dog',
           'horse','sheep','aeroplane','bicycle','boat',
           'bus','car','motorbike','train','bottle',
           'chair','diningtable','pottedplant','sofa','tvmonitor']

total = dict(zip(the_classes, [0]*(len(the_classes))))
well_classified = dict(zip(the_classes, [0]*(len(the_classes))))

i = 0
for image in original.values:
    tmp = dict(zip(the_classes, [0]*(len(the_classes))))
    
    for obj in image[4].values :
        total[obj[0]] = total.get(obj[0]) + 1
        tmp[obj[0]] = tmp.get(obj[0]) + 1
        
    tmp2 = dict(zip(the_classes, [0]*(len(the_classes))))
    for obj_predict in result.values[i][1].values :
        tmp2[obj[0]] = tmp2.get(obj[0]) + 1
        
    for cl in the_classes :
        if tmp.get(cl) != 0 :
            well_classified[obj[0]] = well_classified.get(obj[0]) + min(tmp.get(cl),tmp2.get(cl))

    i += 1
    
for cl in the_classes :
    accuracy = (well_classified.get(cl) / total.get(cl)) * 100
    print("The accuracy of the class ", cl," is : ",accuracy)
    
    
