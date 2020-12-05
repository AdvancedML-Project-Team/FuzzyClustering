import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import os
import time
print(os.getcwd())
# %% 데이터 불러오기 택1 ###########################################################
# %% 연습데이터 클러스터 2개 활모양
df = pd.read_csv('data/2d_dataset_2.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
# %% 사이킷런 링 데이터
df, y = datasets.make_circles(n_samples=400, factor=.1, noise=.1)
# %% 사이킷런 5 blobs 데이터
centers = [[0, 0], [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7]]
df, y = datasets.make_blobs(n_samples=400, n_features=2, centers=centers, cluster_std=0.15)
# %% 불균형데이터
df = pd.read_csv('data/practice.txt', sep=' ', header=None)
df = np.asarray(df)
# %% 여러 군집형 데이터
df = pd.read_csv('data/Aggregation.txt', sep='\t', header=None)
df = df.iloc[:, :-1]
df = np.asarray(df)
# %% 연습데이터 클러스터 2개
df = pd.read_csv('data/2d_dataset_1.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
# %% 연습데이터 클러스터 2개 활모양
df = pd.read_csv('data/2d_dataset_2.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
# %% 연습데이터 클러스터 2개 고리모양
df = pd.read_csv('data/2d_dataset_3.csv')
df = df.iloc[:, 1:]
df = np.asarray(df)
###############################################################################

# %% parameter 세팅 ############################################################
N = df.shape[0]
p = df.shape[1]
plt.scatter(df[:,0], df[:,1])
###############################################################################

# %% 중심점 초기화 및 Kernel함수 정의 ############################################
# 3) Initialize centroids
K = 5
m = 2
init_ctr_idx = np.random.choice(N, K)
ctr = df[init_ctr_idx]


def Kernel(X, Y, b=0.5, d=2):
    if X.ndim > 1 and Y.ndim > 1:
        return (np.dot(X, Y.T) + b) ** d

    return (np.dot(X, Y) + b) ** d

# 4) Cumpute the degree of membership of all feature vectors in
# all the clusters uik


D = np.zeros((N, K))
eps = 10e-5

for i in range(N):
    for k in range(K):
        d_square = Kernel(df[i], df[i]) - 2 * Kernel(df[i], ctr[k]) +\
            Kernel(ctr[k], ctr[k])
        D[i, k] = (1 / (d_square + eps)) ** (1 / (m - 1))
row_sums = D.sum(axis=1)
prev_U = D / row_sums[:, np.newaxis]

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

###############################################################################

# %% 반복구간 #####################################################################
# 5-2) Update the degree of membership uik to uik_hat according to 4)


change = 1
it = 1
while((change > 0.01) and it < 30):
    print('#### iter : {}/{} ####'.format(it, N))
    D = np.zeros((N, K))
    for i in range(N):
        if i % 100 == 0:
            print('i:{}...'.format(i))
        for k in range(K):
            d_square =\
                Kernel(df[i], df[i]) -\
                2 * Kernel_xi_newc(df[i], k, prev_U) +\
                Kernel_newc_newc(k, prev_U)

            D[i, k] = (1 / d_square) ** (1 / (m - 1))
    row_sums = D.sum(axis=1)
    curr_U = D / row_sums[:, np.newaxis]
    change = max(np.linalg.norm(prev_U - curr_U, axis=1))

    print('max(|U - U\'|) : {}'.format(change))
    prev_U = curr_U
    it += 1
###############################################################################

# %% 2차원 데이터 scatter로 확인 #################################################

y_pred = np.argmax(curr_U, axis=1)
plt.scatter(df[:, 0], df[:, 1], c=y_pred)

###############################################################################

# %% k값 바꾸면서 elbow 그리기(오래걸림주의) ######################################
def get_silhouette(df, y_pred):
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

def predict(df, K, m=2, maxiter = 30, epsilon = 0.05):
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
    change = 1
    it = 1
    while((change > epsilon) and it < maxiter):
        print('iter : {}'.format(it))
        D = np.zeros((N, K))
        for i in range(N):
            if i % 100 == 0:
                print('i : {:>3}/{}'.format(i,N))
            for k in range(K):
                d_square =\
                    Kernel(df[i], df[i]) -\
                    2 * Kernel_xi_newc(df[i], k, prev_U) +\
                    Kernel_newc_newc(k, prev_U)
                    
                D[i, k] = (1 / d_square) ** (1 / (m - 1))
        row_sums = D.sum(axis=1)
        curr_U = D / row_sums[:, np.newaxis]
        change = max(np.linalg.norm(prev_U - curr_U, axis=1))
        print()
        print('change : {}'.format(change))
        prev_U = curr_U
        it += 1
    y_pred = np.argmax(curr_U, axis=1)
    
    return y_pred
#%% k값 하나로 예측할 때 ########################################################
K = 4
m = 2
y_pred = predict(df, K=K, m=m)
plt.scatter(df[:, 0], df[:, 1], c=y_pred)

#%% k값 여러개로 그림그릴때(기본설정 k:1~10 ######################################
ks = []
silhouettes = []
y_preds = []
for k in range(1, 10+1):
    print('*************************************k : {}'.format(k))
    y_pred = predict(df, k)
    ks.append(k)
    silhouettes.append(get_silhouette(df, y_pred))
    y_preds.append(y_pred)
    
plt.scatter(ks, silhouettes)
#%% y_preds로 그림그리기 #######################################################
plt.scatter(df[:,0], df[:,1], c=y_preds[0])
plt.scatter(df[:,0], df[:,1], c=y_preds[1])
plt.scatter(df[:,0], df[:,1], c=y_preds[2])
plt.scatter(df[:,0], df[:,1], c=y_preds[3])
plt.scatter(df[:,0], df[:,1], c=y_preds[4])
plt.scatter(df[:,0], df[:,1], c=y_preds[5])
plt.scatter(df[:,0], df[:,1], c=y_preds[6])
plt.scatter(df[:,0], df[:,1], c=y_preds[7])
plt.scatter(df[:,0], df[:,1], c=y_preds[8])
plt.scatter(df[:,0], df[:,1], c=y_preds[9])


































































































































































