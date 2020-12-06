
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from collections import Counter

#%%
df = pd.read_csv('C:/Users/bosya/OneDrive/바탕 화면/ㄱ/wine-clustering.csv')
df = np.asarray(df)
N = df.shape[0]
p = df.shape[1]

#%%

df, y = datasets.make_circles(n_samples=600, factor=.5, noise=.05, random_state=42)
plt.scatter(df[:, 0], df[:, 1])

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

start_weights =  distance_array_(norm_df, start_centroid)


print(start_weights.shape)
print("초기 centroid 개수",len(ini_cent))
print("빈거랑 별로 없는거 지우고 개수",len(centroid_1st))
print("두개씩 다 합치고 남은 최종 시작 개수",len(start_centroid))
print(start_centroid)

#%%
K = len(start_centroid)
N = df.shape[0]

#%%

def update_centroid(weights, K, m=2):
    centroid = []
    for k in range(K):
        wk = weights[:, k]**m
        denominator = df * wk[:, None]
        centroid.append(np.sum(denominator, axis=0) / np.sum(wk))
    return centroid


def update_weight_Gau(centroid):
    w_bunmos = []
    w_bunjas = []
    for i in range(len(centroid)):
        w_bunmo = 0
        bunjas = []
        for xi in norm_df:
            bunja = np.linalg.norm(xi-centroid[i])/norm_df.std()
            w_bunmo += bunja
            bunjas.append(bunja)
        w_bunmos.append(w_bunmo)
        w_bunjas.append(bunjas)
    
    weights = np.zeros([N,K])
    for m in range(len(w_bunmos)):
        for j in range(len(w_bunjas[m])):
            weights[j,m] = w_bunjas[m][j]/w_bunmos[m]
            
    return weights


def predict(df, centroid, weight, K=3, m=2, epsilon=0.01, max_iter=50):

    prev_centroid = centroid

    # 반복구간
    cnt = 1
    change_max = 100
    while (change_max > epsilon) and (max_iter > cnt):
        curr_centroid = update_centroid(weight, K)
        change = []
        for k in range(K):
            change.append(np.linalg.norm(curr_centroid[k] - prev_centroid[k]))
        change_max = max(change)
        weight = update_weight_Gau(curr_centroid)
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
# #%%
# K = len(start_centroid)
# m = 2
# epsilon = 0.01
# max_iter = 50


# y_pred, weight, curr_centroid = predict(norm_df, start_centroid, start_weights, K, m, epsilon = epsilon, max_iter=max_iter)

# plt.scatter(norm_df[:, 0], norm_df[:, 1], c=y_pred)  # 그림확인
# for k in range(K):  # centroid 그리기
#     plt.scatter(curr_centroid[k][0], curr_centroid[k][1], s=100)

# #%%

# m = 2
# epsilon = 0.01
# max_iter = 50

# ks = []
# costs = []
# y_preds = []

# max_K = 20
# for k in range(2, max_K+1):
#     print('K : {:>2}/{}'.format(k, max_K))
#     y_pred, weight, curr_centroid = predict(norm_df, start_centroid, start_weights, K, m, epsilon = epsilon, max_iter=max_iter)
#     ks.append(k)
#     # costs.append(get_cost(df, y_pred, curr_centroid))  # cost 둘중 아무거나 써도됨
#     costs.append(get_silhouette(norm_df, k, y_pred))
#     y_preds.append(y_pred)

# plt.scatter(ks, costs)  # Elbow 그래프
# plt.xlabel('군집 수 K')
# plt.ylabel('Silhouette')
