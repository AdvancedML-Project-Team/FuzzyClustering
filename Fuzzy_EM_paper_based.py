# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:41:08 2020

@author: user
"""

import pandas as pd
import numpy as np
from collections import Counter

#%%
df = pd.read_csv('C:/Users/user/OneDrive/바탕 화면/DS/2nd_sem/AdvancedML/wine-clustering.csv')
df = np.asarray(df)
N = df.shape[0]
p = df.shape[1]
#%%

def normalize_data(df): ## data 정규화
    norm_df = (df - df.mean())/df.std()
    
    return norm_df


def initial_centroid(df, n_cls): ## n_cls 개 만큼의 초기 random seeds 생성
    
    # n_rows = df.shape[0]
    n_cols = df.shape[1]
    
    centroids = np.zeros([n_cls,n_cols])
    
    for k in range(n_cls):
        centroid = []
        for i in range(n_cols):          
            random = np.random.normal(df[:,i].mean(), df[:,i].std(),1)
            centroid.append(random[0])
        centroids[k] = centroid
        
    return centroids

def distance_array_(df, centroids):
    
    distance_array = np.zeros([df.shape[0], centroids.shape[0]])
    
    for i, centroid in enumerate(centroids):
        distances = []
        for xi in df:
            dist = np.linalg.norm(xi-centroid)
            distances.append(dist)
        distance_array[:,i] = distances
    
    return distance_array
    

def delelte_cluster(df, centroids, distance_array): ## 거리기반으로 clustering 후 비어있거나 적은 centroid 제거
    
    labels = []
    for idx in range(len(df)):
        label = np.argmin(distance_array[idx,:])
        labels.append(label)
   
    del_idx = []
    for i, (k, v) in enumerate(Counter(labels).items()):    
        if v < 5: ## 5개보다 적은 datapoint를 가지고있는 centroid 제거
            del_idx.append(i)
    
    new_centroids = np.delete(centroids, del_idx, axis = 0)
    
    new_dist = distance_array_(df, new_centroids)
    
    new_labels = []
    for idx in range(len(df)):
        label = np.argmin(new_dist[idx,:])
        new_labels.append(label)

    return new_centroids, new_labels

def concat_cluster(centroids, thrs): ## 정해진 거리 만큼 가까이 있는 centroid 확인
    result = []
    for i in range(len(centroids)):
        tmp = {}
        for j, centroid in enumerate(centroids):
            dist = np.linalg.norm(centroids[i]-centroid)
            if dist < thrs:
                tmp[j] = dist
        result.append(tmp)
    
    change_dict= {}
    for i, dict_ in enumerate(result):
        del dict_[i]
        change_dict[i] = list(dict_.keys())
    
    
    return change_dict
        

    
        
        

#%% main

norm_df = normalize_data(df)

ini_cent = initial_centroid(norm_df, 50)

distance_array =  distance_array_(norm_df, ini_cent)

centroid_1st, update_labels = delelte_cluster(norm_df, ini_cent, distance_array)

label_dict = Counter(update_labels)

re = concat_cluster(centroid_1st, 1)

print(distance_array)
print("초기 centroid 개수",len(ini_cent))
print("빈거랑 별로 없는거 지우고 개수",len(centroid_1st))

print(re)

