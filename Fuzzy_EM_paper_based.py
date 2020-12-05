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

def concat_two(centroids): ## 가까이 있는 두개의 클러스터 concat
    del_idx = []
    for i in range(len(centroids)):
        tmp = []
        for j, centroid in enumerate(centroids):
            if i != j:
                dist = np.linalg.norm(centroids[i]-centroid)
                tmp.append(dist)
        if min(tmp) < 1.5: ## 거리가 저정도되는 클러스터를 합치려고 하면 멈추게
            del_ = np.argmin(tmp)
            del_idx.append(del_)
            centroids[del_] = centroids[i]
        else:
            break
    
    cent_list = np.array(centroids.tolist())
    
    new_centroids = np.delete(cent_list, del_idx, axis = 0)
    
    return new_centroids

def repeat(new_centroids): ## 두개씩 합치는거 반복 멈출때 concat_two() 함수의 조건 때문에 더이상 합치지 않을때까지
    a = new_centroids
    tmp=[]
    while True:
        a = concat_two(a)
        tmp.append(len(a))
        if len(a) == tmp[-1]:
            break
    return a
         

#%% main

norm_df = normalize_data(df)

ini_cent = initial_centroid(norm_df, 50)

distance_array =  distance_array_(norm_df, ini_cent)

centroid_1st, update_labels = delelte_cluster(norm_df, ini_cent, distance_array)

starting_centroid = concat_two(centroid_1st)

start_centroid = repeat(starting_centroid)

d =  distance_array_(norm_df, start_centroid)

_, start_label = delelte_cluster(norm_df, start_centroid, d)

print("초기 centroid 개수",len(ini_cent))
print("빈거랑 별로 없는거 지우고 개수",len(centroid_1st))
print("두개씩 다 합치고 남은 최종 시작 개수",len(start_centroid))
print(start_centroid)

