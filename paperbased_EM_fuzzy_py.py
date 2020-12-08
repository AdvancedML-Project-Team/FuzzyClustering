# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:49:10 2020

@author: KYEONGCHAN LEE
"""

import math
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from collections import Counter

# %% 사이킷런 링 데이터 (과녁형)
df, y = datasets.make_circles(n_samples=400, factor=.1, noise=.1, random_state=42)
plt.scatter(df[:, 0], df[:, 1])
# %% 사이킷런 링 데이터 (고리 2개)
df, y = datasets.make_circles(n_samples=600, factor=.5, noise=.05, random_state=42)
plt.scatter(df[:, 0], df[:, 1])
# %% 사이킷런 교차 데이터
df, y = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, random_state=(1))
plt.scatter(df[:, 0], df[:, 1])
#%%
centers = [[0, 0], [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7]]
df, y = datasets.make_blobs(n_samples=400, n_features=2, centers=centers, cluster_std=0.15)
plt.scatter(df[:, 0], df[:, 1])
#%%
def random_centroid(df, n_cls):
    n_cols = df.shape[1]
    
    centroids = np.zeros([n_cls,n_cols])
    
    for k in range(n_cls):
        centroid = []
        for i in range(n_cols):          
            random = np.random.normal(df[:,i].mean(), df[:,i].std(),1)
            centroid.append(random[0])
        centroids[k] = centroid
        
    return centroids
#%%
start_centroid = random_centroid(df, 50)

#%%
def visualize(df, centroid):
    plt.scatter(df[:, 0], df[:, 1])
    for k in range(len(centroid)):  # centroid 그리기
        plt.scatter(centroid[k][0], centroid[k][1], s=100)
#%%     
visualize(df, start_centroid)

#%%
def point_to_centroid(centroids):
    distance = np.zeros([df.shape[0],len(centroids)])
    for i, centroid in enumerate(centroids):
        for j, xi in enumerate(df):
            dist = np.linalg.norm(xi-centroid)
            distance[j,i] = dist
    
    return distance

#%%
distance = point_to_centroid(start_centroid)
distance.shape
#%%
def reduce_centroid(centroids, distance):
    belong_to = np.zeros([df.shape[0],len(centroids)])
    
    for j, dist in enumerate(distance):
        for i, d in enumerate(dist):
            if d < 0.5:
                belong_to[j,i] = 1
    del_idx = []            
    for k in range(len(centroids)):
        if Counter(belong_to[:,k])[0] > len(df)-5: ## 클러스터에 속한 포인트가 10개 이하면 삭제
            del_idx.append(k)
            
    new_centroids = np.delete(centroids, del_idx, axis = 0)
    
    return new_centroids

reduced_centroids = reduce_centroid(start_centroid, distance)
#%%
visualize(df, reduced_centroids)
#%%
def concat_centroid(centroids):
    del_idx = []
    i=0
    while i == len(centroids):
        tmp = []
        for j in range(len(centroids)):
            if i < j:
                c_dist = np.linalg.norm(centroids[i] - centroids[j])
                tmp.append(c_dist)
            else :
                tmp.append(100)
          
        while min(tmp) < 2:
            del_idx.append(np.argmin(tmp))
            tmp[np.argmin(tmp)] = 100 
        
        i+=1
        

    
    new_centroids = np.delete(centroids, del_idx, axis = 0)
          
    return new_centroids



def concat_centroid(centroids):
    del_idx = []
    for i in range(len(reduced_centroids)-1):
        tmp = []
        for j in range(len(centroids)):
            if i < j:
                c_dist = np.linalg.norm(centroids[i] - centroids[j])
                tmp.append(c_dist)
            else :
                tmp.append(100)
#         if min(tmp) < 0.5:
                del_idx.append(np.argmin(tmp))
    
    new_centroids = np.delete(centroids, del_idx, axis = 0)
          
    return new_centroids

#%%
new_centroids = concat_centroid(reduced_centroids)

while len(new_centroids) > 6:
    new_centroids = concat_centroid(new_centroids)
#%% 
result = new_centroids
#%%
result
#%%
visualize(df, result)

#%%
def tagging(centroids):
    
    distances = point_to_centroid(centroids)
    labels = []
    for dist in distances:
        labels.append(np.argmin(dist))
        
    return labels
#%%
tag_ = tagging(result)

plt.scatter(df[:, 0], df[:, 1], c=tag_)
for k in range(len(result)):  # centroid 그리기
    plt.scatter(result[k][0], result[k][1], s=100)

#%%
def update_centroid(centroids, tag_):
    data = pd.DataFrame(df)
    data['tag'] = tag_
    data_2 = data.sort_values('tag')
    data_2 = data_2.reset_index()
    centroids_list = []
    for j in range(len(centroids)):
        asd = data_2[data_2['tag']==j]
        del asd['index']
        del asd['tag']

        summation = 0
        data = []
        for i in range(len(asd)):
            xi = [asd.iloc[i,0], asd.iloc[i,1]]
            data.append(xi)
            x_mu = xi-centroids[j]
            tmp = np.sqrt(np.square(x_mu[0]) + np.square(x_mu[1])) 
            summation += math.exp((tmp)/2*np.square(df.std()))
        weight = []
        for i in range(len(asd)):
            xi = [asd.iloc[i,0], asd.iloc[i,1]]
            x_mu = xi-centroids[j]
            tmp = np.sqrt(np.square(x_mu[0]) + np.square(x_mu[1]))
            weight.append(math.exp((tmp)/2*np.square(df.std())) / summation)

        centroids_list.append(sum(np.array(data) * np.array(weight)[:,None]))

    return centroids_list
#%%
updata_cent = update_centroid(result, tag_)

#%%
updata_cent

#%%
visualize(df, updata_cent)
#%%
re = concat_centroid(updata_cent)
#%%
re
#%%
visualize(df, re)
#%%
up_tag = tagging(re)
#%%
updata_2 = update_centroid(re, up_tag)
#%%
visualize(df, updata_2)
#%%
plt.scatter(df[:, 0], df[:, 1], c=up_tag)
for k in range(len(updata_2)):  # centroid 그리기
    plt.scatter(updata_2[k][0], updata_2[k][1], s=100)

#%%
up_tag_2 = tagging(updata_2)
updata_3 = update_centroid(updata_2, up_tag_2)

plt.scatter(df[:, 0], df[:, 1], c=up_tag_2)
for k in range(len(updata_3)):  # centroid 그리기
    plt.scatter(updata_3[k][0], updata_3[k][1], s=100)
























