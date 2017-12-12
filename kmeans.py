#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:39:55 2017

KMeans implementation

@author: jmarnat
"""

import numpy as np
import sys

def is_there_doubles(l):
    for i in range(len(l)):
        for j in range(i+1,len(l)):
            if l[i] == l[j]:
                return True
    return False

def kmeans_centroids(X, n, eps, max_iter):
    if n > len(X):
        print('Error: n > len(X)')
        sys.exit(0)
    
    # n centroids initialization (just once)
    np.random.seed(0)
    while True:
        r = [np.random.randint(0,len(X)) for i in range(n)]
        if not is_there_doubles(r):
            break
    
    #todo check r for doubles
    centroids_before = X[r]
    
    i_iter = 0
    while True:
        # 
        y_c = kmeans_fit(X,centroids_before)
        
        centroids_after = np.zeros((n,len(X[0])))
        sum_moove = 0
        for c in range(n):
            for d in range(np.size(X,axis=1)):
                centroids_after[c,d] = np.mean(X[y_c==c,d])
            # adding the distance from the old centroid to the new one
            sum_moove += np.linalg.norm(centroids_before[c] - centroids_after[c])
    
        if (i_iter >= max_iter): break
        if (sum_moove <= eps): break
        
        ++i_iter
        centroids_before = centroids_after
        
    return centroids_after

def kmeans_fit(X,centroids):
    y = np.zeros(len(X))
    dist = np.zeros(len(centroids))
    for i in range(len(X)):
        # computing the argmin distance to each centroid
        for c in range(len(centroids)):
            dist[c] = np.linalg.norm(X[i] - centroids[c])
        y[i] = np.argmin(dist)
        
    return y


















