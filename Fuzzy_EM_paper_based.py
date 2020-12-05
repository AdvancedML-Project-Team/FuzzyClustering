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
print(df)

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

def delelte_empty_cluster(df, centroids): ## 거리기반으로 clustering 후 비어있는 centroid 제거
    
    distance_array = np.zeros([df.shape[0], centroids.shape[0]])
    
    for i, centroid in enumerate(centroids):
        distances = []
        for xi in df:
            dist = np.linalg.norm(xi-centroid)
            distances.append(dist)
        distance_array[:,i] = distances
    
    labels = []
    
    for idx in range(len(df)):
        label = np.argmin(distance_array[idx,:])
        labels.append(label)
   
    del_idx = []
    for i, (k, v) in enumerate(Counter(labels).items()):    
        if v == 0:
            del_idx.append(i)
    
    new_centroids = np.delete(centroids, del_idx, axis = 0)

    return new_centroids, labels, distance_array

# 딱히 알필요 없는듯
# def belong_to_where(centroids, distances, thrs): ## 각 포인트가 어느 클러스에 속하는지 가까운 데로 정렬되 리스트 생성
    
#     belong_to_cls = []
#     for dist in distances:
#         tmp = []
#         d = np.argmin(dist)
#         tmp.append(d)
#         while dist[d] < thrs:
#             dist[d] = 10
#             d = np.argmin(dist)
#             tmp.append(d)
#         belong_to_cls.append(tmp)
        
#     return belong_to_cls
            
def concat_cluster(centroids, thrs): ## 정해진 거리 만큼 가까이 있는 centroid 확인
    result = []
    for i in range(len(centroids)):
        tmp = []
        for j, centroid in enumerate(centroids):
            dist = np.linalg.norm(centroids[i]-centroid)
            if dist < thrs:
                tmp.append(j)
        result.append(tmp)
    
    return result
        


        
    
        
        

#%% main

norm_df = normalize_data(df)

ini_cent = initial_centroid(norm_df, 10)

centroid_1st, labels, distances = delelte_empty_cluster(norm_df, ini_cent)

label_dict = Counter(labels)

# clst = belong_to_where(centroid_1st, distances, 1)

re = concat_cluster(centroid_1st, 0.5)

print(distances)
print("초기 centroid 개수",len(ini_cent))
print("빈값지우고 개수",len(centroid_1st))
# print(clst)
print(re)

