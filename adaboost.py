#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from os import chdir
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

chdir('/home/jmarnat/Documents/CV-project/cvp')



X_train = pd.read_csv('X_surf-5000.csv')
X_train = X_train.drop('Unnamed: 0',axis=1)
X_train = np.array(X_train)

y_train = pd.read_csv('y_surf-5000.csv')
y_train = y_train.drop('Unnamed: 0',axis=1)
y_train = y_train.values
y_train = np.asarray(y_train)
#y_train = np.ones(len(y_train))
y_train = y_train.astype(str)


            
classes = ['person','bird','cat','cow','dog',
           'horse','sheep','aeroplane','bicycle','boat',
           'bus','car','motorbike','train','bottle',
           'chair','diningtable','pottedplant','sofa','tvmonitor']

classes = pd.DataFrame(classes)
classes = classes.rename(columns={0:'class'})

for y_i in range(len(y_train)):
    y_train[y_i] = np.where(classes==y_train[y_i])[0][0]
y_train = y_train.astype(int)


# train // test
X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.33, random_state=42)




bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
                         algorithm="SAMME",
                         n_estimators=200)


time_a = datetime.now()
bdt.fit(X_train, y_train.ravel())
time_b = datetime.now()

print('started at    ',time_a)
print('ended at      ',time_b)
print('fitting-time:',time_b - time_a) 

#res = bdt.predict([X_test[0]])[0]
#res

time_a = datetime.now()

results = pd.DataFrame()
for i in range(len(X_test)):
    print(np.round(100*i/len(X_test)),"/100")
    class_i = y_test[i][0]
    x_test_i = X_test[i]
    pred = bdt.predict([X_test[i]])[0]
    newrow = pd.DataFrame([class_i,pred]).T
    results = results.append(newrow)

time_b = datetime.now()
print('started at    ',time_a)
print('ended at      ',time_b)
print('fitting-time: ',time_b - time_a)





            
the_classes = ['person','bird','cat','cow','dog',
           'horse','sheep','aeroplane','bicycle','boat',
           'bus','car','motorbike','train','bottle',
           'chair','diningtable','pottedplant','sofa','tvmonitor']

results2 = results
#results2 = pd.read_csv('results-sifts-500.csv')
results2.columns = ['y','pred']

mean = 0
for i in range(20):
    acc = accuracy_score(
        results2[results2['y']==i]['y'],
        results2[results2['y']==i]['pred'])
    mean += acc
    print('class=',the_classes[i],'\t\taccuracy=',acc)
mean /= 20
print('mean acc = ',mean)





