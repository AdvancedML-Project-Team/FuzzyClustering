# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 02:44:40 2020

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
#%% get_centroid(weight)
def get_centroid(weight, K):
    centroid = []
    for k in range(K):
        wk = weight[:,k] ** m
        denominator = df * wk[:, None]
        centroid.append(np.sum(denominator, axis=0) / np.sum(wk))
    return centroid
#%% def update_weight(centroid)
def update_weight(centroid, N, K):
    weight = np.zeros((N,K))
    for i in range(N):
        for j in range(K):
            xi = df[i]
            summation = 0
            for k in range(K):
                summation += (np.linalg.norm(xi-centroid[j]) / np.linalg.norm(xi-centroid[k])) ** (2/(m-1))
            weight[i, j] = 1 / summation
    
    return weight

#%% 반복구간


def predict(df, K=3, m=2, epsilon=0.05, max_iter=50):
    print(K)
    N = df.shape[0]
    weight = np.random.rand(N, K)
    weight /= weight.sum(axis=1, keepdims=1)
    prev_centroid = []
    for k in range(K):
        prev_centroid.append(df[k])
    cnt = 1
    change = [100] * K

    while (sum(np.asarray(change) > epsilon) > 0) or (max_iter >= cnt):
        # print('iter:{}'.format(cnt))
        curr_centroid = get_centroid(weight, K)
        change = []
        for k in range(K):
            change.append(np.linalg.norm(curr_centroid[k] - prev_centroid[k]))
        # print('center change:{}'.format(change))
        weight = update_weight(curr_centroid, N, K)
        prev_centroid = curr_centroid
        cnt += 1
    y_pred = np.argmax(weight, axis=1)
    
    return y_pred, weight, curr_centroid

#%% cost 구하기


def get_cost(df, y_pred, curr_centroid):
    N = df.shape[0]
    p = df.shape[1]
    clusters = np.zeros((N, p))
    for i in range(N):
        cluster = y_pred[i]
        clusters[i] = curr_centroid[cluster]
    cost = np.sum(np.linalg.norm(df - clusters, axis = 1) ** 2)
    
    return cost
#%% 5000 by 2 연습데이터 클러스터 15개
df = pd.read_csv('data/S-sets.csv', header=None)
df = np.asarray(df)
N = df.shape[0]
p = df.shape[1]
plt.scatter(df[:,0], df[:,1])

#%% wine 데이터 
df = pd.read_csv('data/wine-clustering.csv')
df = np.asarray(df)
N = df.shape[0]
p = df.shape[1]
#%% iris 데이터
iris = datasets.load_iris()
df = pd.DataFrame(iris['data'])
df = np.asarray(df)
N = df.shape[0]
p = df.shape[1]
#%% 연습데이터 클러스터 2개
df = pd.read_csv('data/2d_dataset_1.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
N = df.shape[0]
p = df.shape[1]
plt.scatter(df[:,0], df[:,1])
#%% 연습데이터 클러스터 2개 활모양
df = pd.read_csv('data/2d_dataset_2.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
N = df.shape[0]
p = df.shape[1]
plt.scatter(df[:,0], df[:,1])
#%% 여러 군집형 데이터

df = pd.read_csv('data/Aggregation.txt', sep='\t', header=None)
df = df.iloc[:, :-1]
df = np.asarray(df)
N = df.shape[0]
p = df.shape[1]
plt.scatter(df[:,0], df[:,1])

#%% 3차원 이상 pairplot으로 확인
plot_df = pd.DataFrame(df)
plot_df['y'] = y_pred
sns.pairplot(plot_df, hue='y')
#%% 2차원 scatter로 확인
plt.scatter(df[:, 0], df[:, 1], c=y_pred)
for k in range(K):
    plt.scatter(curr_centroid[k][0], curr_centroid[k][1], s=100)
#%%

plt.scatter(df[:,0], df[:,1])
K = 3
m = 2
epsilon = 0.05
max_iter = 50

ks = []
costs = []
for k in range(1, 10+1):
    y_pred, weight, curr_centroid = predict(df, K=k)
    ks.append(k)
    costs.append(get_cost(df, y_pred, curr_centroid))
#%%
plt.scatter(ks, costs)


