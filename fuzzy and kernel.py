import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
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

# %% 함수 정의 #################################################################


def Kernel(X, Y, b=.5, d=2):
    if X.ndim > 1 and Y.ndim > 1:
        return (np.dot(X, Y.T) + b) ** d

    return (np.dot(X, Y) + b) ** d

# 5-1) Compute new kernel matrix K(Xi, Vk) and K(Vj_hat, Vj_hat)


def Kernel_xi_newc(Xi, k, U):
    mth_powerd = U[:, k] ** m

    return np.sum(mth_powerd * Kernel(df, Xi)) / np.sum(mth_powerd)


def Kernel_newc_newc(k, U):
    mth_powerd = U[:, k] ** m
    denominator = np.sum(np.dot(mth_powerd.reshape((-1, 1)),
                                mth_powerd.reshape((1, -1))) * Kernel(df, df))
    numerator = np.sum(mth_powerd) ** 2

    return denominator / numerator


# silhouette 구하기 함수
def get_silhouette(df, K, y_pred):
    N = df.shape[0]
    p = df.shape[1]

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


# 예측 함수
def predict(df, K, m=2, maxiter=30, epsilon=0.01):
    N = df.shape[0]
    p = df.shape[1]

    # 3) Initialize centroids
    init_ctr_idx = np.random.choice(N, K)
    ctr = df[init_ctr_idx]

    # 4) Cumpute the degree of membership of all feature vectors in all the clusters uik
    D = np.zeros((N, K))
    eps = 10e-5

    for i in range(N):
        for k in range(K):
            d_square = Kernel(df[i], df[i]) - 2 * Kernel(df[i], ctr[k]) +\
                Kernel(ctr[k], ctr[k])
            D[i, k] = (1 / (d_square + eps)) ** (1 / (m - 1))
    row_sums = D.sum(axis=1)
    prev_U = D / row_sums[:, np.newaxis]

    # 5-2) Update the degree of membership uik to uik_hat according to 4)
    # 반복구간
    change = 1
    it = 1
    while((change > epsilon) and it < maxiter):
        print('iter : {}'.format(it))
        D = np.zeros((N, K))
        for i in range(N):
            if i % 100 == 0:
                print('i : {:>5}/{}'.format(i, N))
            for k in range(K):
                d_square =\
                    Kernel(df[i], df[i]) -\
                    2 * Kernel_xi_newc(df[i], k, prev_U) +\
                    Kernel_newc_newc(k, prev_U)

                D[i, k] = (1 / d_square) ** (1 / (m - 1))
        row_sums = D.sum(axis=1)
        curr_U = D / row_sums[:, np.newaxis]
        change = max(np.linalg.norm(prev_U - curr_U, axis=1))
        print('U change : {}'.format(change))
        print()
        prev_U = curr_U
        it += 1
    y_pred = np.argmax(curr_U, axis=1)

    return y_pred
#%% k값 하나로 예측할 때 ########################################################

K = 2
m = 2
y_pred = predict(df, K=K, m=m)
plt.scatter(df[:, 0], df[:, 1], c=y_pred)
plt.figtext(.5,.9,'Fuzzy+Kernel clustering', fontsize=20, ha='center')
plt.text(0.85, 1, s='d=2', fontsize=15)
get_silhouette(df, K, y_pred)
###############################################################################

#%% k값 여러개로 그림그릴때(기본설정 k:2~10) #####################################

m = 2
ks = []
silhouettes = []
y_preds = []
for k in range(2, 10+1):
    print('*************************************k : {}'.format(k))
    y_pred = predict(df, k)
    ks.append(k)
    silhouettes.append(get_silhouette(df, k, y_pred))
    y_preds.append(y_pred)

plt.scatter(ks, silhouettes) #  Elbow 그래프
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
###############################################################################
