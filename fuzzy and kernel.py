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

#%%
K = 5
init_ctr_idx = np.random.choice(N, K)
ctr = df[init_ctr_idx]

#%%
def get_weight():

a = np.array([[1,2, 3],
             [1,3, 6]])
b = np.array([[1, 2 ,1],
              [3, 4, 6]])
np.linalg.norm(a-b, axis=1)

U = np.zeros((N, K))
m = 2
for k in range(K):
    rep_ctr = np.tile(ctr[k],(N,1))
    d = np.linalg.norm(df - rep_ctr, axis=1)
    denominator = (1 / (d ** 2)) ** (1 / (m - 1))
    U[:,k] = denominator

row_sums = U.sum(axis=1)
U = U / row_sums[:, np.newaxis]


















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




















































































































































































































































































































































































































































































































