import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors  
from scipy.spatial import distance

centers = np.load('centers_2_kmeans.npy')

dado = [0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,25,3,0,0,0,0,0,0,1,0,0,0,0,0,5,3,2,2,0,0,0,0,96,47,49,8,2,2,2,16,10,17,135,92,39,25,5,6,0,0,12,16,4,3,3,0,9,6,8,19,9,7,30,14,114,27,11,12,9,36,93,126,10,4,5,13,41,135,135,57,0,0,1,2,10,80,76,25,2,0,0,5,12,72,87,27,0,0,0,0,4,135,135,14,7,0,0,0,0,96,135,126,5,0,0,0,26,86,82,102]


def cal_dist_and_return_gh(point:list, centers:list):
    dist_list = []
    for i in range(2):
        dist_list.append(distance.euclidean(point, centers[i]))
    
    return dist_list.index(min(dist_list)) + 1


gh = cal_dist_and_return_gh(dado, centers=centers)
print(gh)

dataset = pd.read_csv(f'./inserts_p{gh}_of_2.csv')
dataset = dataset.drop(['gh'], axis=1)
# print(dataset.head(2))

cov = np.cov(dataset.values.T)
inv_covmat = np.linalg.inv(cov)
print('foi')
neigh = NearestNeighbors(n_neighbors= 3, algorithm='brute', metric='mahalanobis', metric_params={'VI': inv_covmat})
print('foi 2')
neigh.fit(dataset.values)
print('foi 3')
dist, index = neigh.kneighbors([dado], 2, True)
#print(type(index[0][0]))
print('foi 4')
print(dataset.loc[index[0][0],:])

