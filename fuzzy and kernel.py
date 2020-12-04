# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 02:44:40 2020

@author: Admin
"""

"""
논문대로 구현하였음
"""

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
print(os.getcwd())
#%% 연습데이터 클러스터 2개 활모양
df = pd.read_csv('data/2d_dataset_2.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
N = df.shape[0]
p = df.shape[1]
plt.scatter(df[:,0], df[:,1])

#%% 사이킷런 링 데이터
df, y = datasets.make_circles(n_samples=400, factor=.1, noise=.1)
N = df.shape[0]
p = df.shape[1]
plt.scatter(df[:,0], df[:,1])
# 3) Initialize centroids
K = 2
m = 2
init_ctr_idx = np.random.choice(N, K)
ctr = df[init_ctr_idx]

#
def Kernel(X, Y, b=0.5, d=10):
    
    if X.ndim > 1 and Y.ndim > 1:
        return (np.dot(X, Y.T) + b) ** d
    
    return (np.dot(X, Y) + b) ** d
# 4) Cumpute the degree of membership of all feature vectors in all the clusters uik

D = np.zeros((N, K))
eps = 10e-5
for i in range(N):
    for k in range(K):
        d_square = Kernel(df[i], df[i]) - 2 * Kernel(df[i], ctr[k]) + Kernel(ctr[k], ctr[k])
        D[i, k] = (1 / (d_square + eps)) ** (1 / (m - 1))
row_sums = D.sum(axis=1)
prev_U = D / row_sums[:, np.newaxis]
print(np.sum(prev_U, axis=1))
# 5-1) Compute new kernel matrix K(Xi, Vk) and K(Vj_hat, Vj_hat)

def Kernel_xi_newc(Xi, k, U):
    mth_powerd = U[:, k] ** m
    return np.sum(mth_powerd * Kernel(df, Xi)) / np.sum(mth_powerd)

def Kernel_newc_newc(k, U):
    mth_powerd = U[:, k] ** m
    denominator = np.sum(np.dot(mth_powerd.reshape((-1, 1)), mth_powerd.reshape((1, -1))) * Kernel(df, df))
    numerator = np.sum(mth_powerd) ** 2

    return denominator / numerator

# Kernel(df, df)
# np.dot(df, df)



# a = mth_powerd = U[:,0] ** m
# b = a
# a = a.reshape((-1, 1))
# b = a.reshape((1,-1))
# c = np.dot(a,b)






# Kernel_matrix = np.zeros((N, N))
# Kernel_matrix
# 676 * 676

# n = 100
# num_runs = 1000

# arr1 = np.random.rand(n)
# arr2 = np.random.rand(n)

# start = time.time()
# arr1_weights = arr2[::-1].cumsum()[::-1] - arr2
# sum_prods = arr1.dot(arr1_weights)
# print("time :", time.time() - start)

# import time
# start = time.time()
# # for r in range(num_runs):
# sum_prod = 0.0
# for i in range(n):
#     for j in range(i+1, n):
#         sum_prod += arr1[i]*arr2[j]
# print("time :", time.time() - start)
    return 2
#%% 5-2) Update the degree of membership uik to uik_hat according to 4)
# while():
change = 1
it = 1
while((change > 0.03) and it < 30):
    D = np.zeros((N, K))
    for i in range(N):
        if i % 50 == 0:
            print('i:{}'.format(i))
        for k in range(K):
            d_square = Kernel(df[i], df[i]) - 2 * Kernel_xi_newc(df[i], k, prev_U) + Kernel_newc_newc(k, prev_U)
            D[i, k] = (1 / d_square) ** (1 / (m - 1))
    row_sums = D.sum(axis=1)
    curr_U = D / row_sums[:, np.newaxis]
    change = max(np.linalg.norm(prev_U - curr_U, axis=1))
    
    print(change)
    prev_U = curr_U
    it += 1

#%% 2차원 scatter로 확인

y_pred = np.argmax(curr_U, axis=1)
plt.scatter(df[:, 0], df[:, 1], c=y_pred)

#%%불균형데이터

df = pd.read_csv('data/practice.txt', sep=' ', header=None)
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
#%% 연습데이터 클러스터 2개 고리모양
df = pd.read_csv('data/2d_dataset_3.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
N = df.shape[0]
p = df.shape[1]
plt.scatter(df[:,0], df[:,1])




















































































































































































































































































































































































































































































































