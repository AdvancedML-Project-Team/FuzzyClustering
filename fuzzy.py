import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
print(os.getcwd())
# %% 데이터 불러오기 택1 ########################################################
# %% 연습데이터 클러스터 2개 활모양
df = pd.read_csv('data/2d_dataset_2.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
plt.scatter(df[:, 0], df[:, 1])
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
# %% 사이킷런 5 blobs 데이터
centers = [[0, 0], [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7]]
df, y = datasets.make_blobs(n_samples=400, n_features=2, centers=centers, cluster_std=0.15)
plt.scatter(df[:, 0], df[:, 1])
# %% 불균형데이터
df = pd.read_csv('data/practice.txt', sep=' ', header=None)
df = np.asarray(df)
plt.scatter(df[:, 0], df[:, 1])
# %% 여러 군집형 데이터
df = pd.read_csv('data/Aggregation.txt', sep='\t', header=None)
df = df.iloc[:, :-1]
df = np.asarray(df)
plt.scatter(df[:, 0], df[:, 1])
# %% 연습데이터 클러스터 2개
df = pd.read_csv('data/2d_dataset_1.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
plt.scatter(df[:, 0], df[:, 1])
# %% 연습데이터 클러스터 2개 활모양
df = pd.read_csv('data/2d_dataset_2.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
plt.scatter(df[:, 0], df[:, 1])
# %% 연습데이터 클러스터 2개 고리모양
df = pd.read_csv('data/2d_dataset_3.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
plt.scatter(df[:, 0], df[:, 1])
###############################################################################

#%% 함수정의 ###################################################################

# get_centroid(weight)
def get_centroid(weight, K):
    centroid = []
    for k in range(K):
        wk = weight[:, k] ** m
        denominator = df * wk[:, None]
        centroid.append(np.sum(denominator, axis=0) / np.sum(wk))
    return centroid


# def update_weight(centroid)
def update_weight(centroid, N, K):
    weight = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            xi = df[i]
            summation = 0
            for k in range(K):
                summation += (np.linalg.norm(xi-centroid[j]) / np.linalg.norm(xi-centroid[k])) ** (2/(m-1))
            weight[i, j] = 1 / summation

    return weight


# 예측함수
def predict(df, K=3, m=2, epsilon=0.01, max_iter=50):
    N = df.shape[0]
    weight = np.random.rand(N, K)
    weight /= weight.sum(axis=1, keepdims=1)
    prev_centroid = []
    for k in range(K):
        prev_centroid.append(df[k])

    # 반복구간
    cnt = 1
    change_max = 100
    while (change_max > epsilon) and (max_iter > cnt):
        curr_centroid = get_centroid(weight, K)
        change = []
        for k in range(K):
            change.append(np.linalg.norm(curr_centroid[k] - prev_centroid[k]))
        change_max = max(change)
        weight = update_weight(curr_centroid, N, K)
        prev_centroid = curr_centroid
        cnt += 1
    y_pred = np.argmax(weight, axis=1)

    return y_pred, weight, curr_centroid


# cost(SSE) 구하기
def get_cost(df, y_pred, curr_centroid):
    N = df.shape[0]
    p = df.shape[1]
    clusters = np.zeros((N, p))
    for i in range(N):
        cluster = y_pred[i]
        clusters[i] = curr_centroid[cluster]
    cost = np.sum(np.linalg.norm(df - clusters, axis=1) ** 2)

    return cost


# silhouette 구하기 함수
def get_silhouette(df, K, y_pred):
    N = df.shape[0]

    si_sum = 0
    for i in range(N):
        samples_in_cluster = df[np.argwhere(y_pred == y_pred[i]).ravel()]
        ai = np.mean(np.linalg.norm(df[i] - samples_in_cluster, axis=1))
        bis = []
        for k in [_ for _ in range(K) if _ != y_pred[i]]:
            samples_in_other_cluster = df[np.argwhere(y_pred == k).ravel()]
            bis.append(np.mean(np.linalg.norm(df[i] - samples_in_other_cluster, axis=1)))
        bi = min(bis)
        si_sum += (bi - ai) / max(ai, bi)

    return si_sum / N
###############################################################################

#%% k한개돌릴때 ################################################################

K = 2
m = 2
epsilon = 0.01
max_iter = 50
y_pred, weight, curr_centroid = predict(df, K, epsilon=epsilon, max_iter=max_iter)

plt.scatter(df[:, 0], df[:, 1], c=y_pred)  # 그림확인
for k in range(K):  # centroid 그리기
    plt.scatter(curr_centroid[k][0], curr_centroid[k][1], s=100)
###############################################################################

#%% k 여러개 돌릴때(기본 2~20) ##################################################

m = 2
epsilon = 0.01
max_iter = 50

ks = []
costs = []
y_preds = []

max_K = 20
for k in range(2, max_K+1):
    print('K : {:>2}/{}'.format(k, max_K))
    y_pred, weight, curr_centroid = predict(df, k, epsilon=epsilon, max_iter=max_iter)
    ks.append(k)
    # costs.append(get_cost(df, y_pred, curr_centroid))  # cost 둘중 아무거나 써도됨
    costs.append(get_silhouette(df, k, y_pred))
    y_preds.append(y_pred)

plt.scatter(ks, costs)  # Elbow 그래프
plt.xlabel('군집 수 K')
plt.ylabel('Silhouette')
###############################################################################

#%% (여러개그리기 연속)y_preds로 그림그리기 ######################################
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[0])  # k=2
plt.title('K = 2')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[1])  # k=3
plt.title('K = 3')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[2])  # k=4
plt.title('K = 4')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[3])  # k=5
plt.title('K = 5')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[4])  # k=6
plt.title('K = 6')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[5])  # k=7
plt.title('K = 7')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[6])  # k=8
plt.title('K = 8')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[7])  # k=9
plt.title('K = 9')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[8])  # k=10
plt.title('K = 10')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[9])  # k=11
plt.title('K = 11')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[10])  # k=12
plt.title('K = 12')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[11])  # k=13
plt.title('K = 13')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[12])  # k=14
plt.title('K = 14')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[13])  # k=15
plt.title('K = 15')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[14])  # k=16
plt.title('K = 16')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[15])  # k=17
plt.title('K = 17')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[16])  # k=18
plt.title('K = 18')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[17])  # k=19
plt.title('K = 19')
#%%
plt.scatter(df[:, 0], df[:, 1], c=y_preds[18])  # k=20
plt.title('K = 20')
###############################################################################
