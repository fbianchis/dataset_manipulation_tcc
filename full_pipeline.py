import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors  
from scipy.spatial import distance
from pyhive import hive
import time

DIVISAO = 1
TYPE_ORGANIZATION = 0
ACTUAL_METRIC = 0
DATA_TO_FIND = 0
FILE_ORGANIZATION = ['Agrupado','Separado por nós']
DATABASE_NAME = ['dataset_full','dataset_partition_4']
TABLE_NAME = ['dataset_full_table','dataset_partition_table']
METRICS = ['l1', 'l2', 'canberra', 'mahalanobis']

data_to_find_neighbors = [[0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,25,3,0,0,0,0,0,0,1,0,0,0,0,0,5,3,2,2,0,0,0,0,96,47,49,8,2,2,2,16,10,17,135,92,39,25,5,6,0,0,12,16,4,3,3,0,9,6,8,19,9,7,30,14,114,27,11,12,9,36,93,126,10,4,5,13,41,135,135,57,0,0,1,2,10,80,76,25,2,0,0,5,12,72,87,27,0,0,0,0,4,135,135,14,7,0,0,0,0,96,135,126,5,0,0,0,26,86,82,102],
[11,45,39,84,27,0,0,0,56,117,56,2,1,1,6,8,37,52,117,22,0,0,7,20,9,30,117,62,0,1,3,2,24,86,22,35,46,1,0,0,44,86,15,6,84,51,20,26,117,25,10,0,9,12,18,116,70,15,45,43,1,0,3,67,49,52,14,17,27,9,2,6,26,68,28,47,117,25,0,2,117,117,15,5,7,4,4,34,110,26,1,0,0,12,23,41,58,6,0,0,17,25,2,6,52,26,9,6,28,52,9,29,28,32,6,0,5,86,47,62,25,3,0,0,10,61,40,90],
[0,1,0,0,1,15,10,3,29,9,0,0,0,28,27,4,145,145,5,0,0,2,5,4,15,135,12,0,3,13,0,0,0,0,0,4,13,28,8,2,91,9,0,3,6,24,15,18,145,51,0,0,0,2,1,49,35,11,0,0,27,111,2,9,4,0,0,5,38,24,14,34,117,30,1,1,6,12,8,11,145,57,0,0,5,42,4,21,22,1,0,0,38,145,4,5,1,0,0,2,7,3,62,113,116,60,1,1,1,1,17,77,145,54,0,0,11,92,12,15,1,0,0,0,10,57,1,0]]



def cal_dist_and_return_gh(point:list, centers:list):
    dist_list = []
    for i in range(DIVISAO):
        dist_list.append(distance.euclidean(point, centers[i]))
    
    return dist_list.index(min(dist_list)) + 1


times = {
    'aplicate_l2':0,
    'get_data_from_database':0,
    'fit_knn':0,
    'calculate_neighborhood':0
}
kmeans_centers = np.load(f'./centers_{DIVISAO}_kmeans.npy')


st = time.time()
gh = cal_dist_and_return_gh(data_to_find_neighbors[DATA_TO_FIND], kmeans_centers)
et = time.time()
times['aplicate_l2'] = et - st
print('ETAPA 1')

conn = hive.Connection(host="192.168.1.23", port=10000, username="username")
st = time.time()
df = pd.read_sql(f"SELECT * FROM {DATABASE_NAME[TYPE_ORGANIZATION]}.{TABLE_NAME[TYPE_ORGANIZATION]} where gh == {gh}", conn)
df = df.drop([f'{TABLE_NAME[TYPE_ORGANIZATION]}.gh'], axis=1)
df = df.values
et = time.time()
conn.close()
times['get_data_from_database'] = et - st
print('ETAPA 2')


st = time.time()
step = int(df.shape[0]/5)
sub = np.array([])
for i in range (0, 2*step, step):
    print(f'INI {i} FIM {min(df.shape[0],i+step)}')
    if METRICS[ACTUAL_METRIC] == 'mahalanobis':
        cov = np.cov(np.transpose(df[i:min(df.shape[0],i+step)][::]))
        try:
            inv_covmat = np.linalg.inv(cov)
        except:
            inv_covmat = np.linalg.pinv(cov)
        neigh = NearestNeighbors(n_neighbors= 3, algorithm='brute', metric=METRICS[ACTUAL_METRIC], metric_params={'VI': inv_covmat})
    else:
        neigh = NearestNeighbors(n_neighbors= 3, algorithm='brute', metric=METRICS[ACTUAL_METRIC])
    neigh.fit(df[i:min(df.shape[0],i+step)][::])
    dist, index = neigh.kneighbors([data_to_find_neighbors[DATA_TO_FIND]], 3, True)
    print('------------')
    print(f'LEN {len(df[i:min(df.shape[0],i+step)][::])} INDEX {index[0]}')
    #print(f'{data[i:min(data.shape[0],i+step)][::]}')
    #print(f'{data[i:min(data.shape[0],i+step)][::][index[0]]}')
    if not i:
        sub = df[i:min(df.shape[0],i+step)][::][index[0]].copy()
    else:
        sub = np.vstack((sub, df[i:min(df.shape[0],i+step)][::][index[0]]))

if METRICS[ACTUAL_METRIC] == 'mahalanobis':
    cov = np.cov(np.transpose(sub))
    try:
        inv_covmat = np.linalg.inv(cov)
    except:
        inv_covmat = np.linalg.pinv(cov)
    neigh = NearestNeighbors(n_neighbors= 3, algorithm='brute', metric=METRICS[ACTUAL_METRIC], metric_params={'VI': inv_covmat})
else:
    neigh = NearestNeighbors(n_neighbors= 3, algorithm='brute', metric=METRICS[ACTUAL_METRIC])
neigh.fit(sub)
dist, index = neigh.kneighbors([data_to_find_neighbors[DATA_TO_FIND]], 3, True)

et = time.time()
times['fit_knn'] = et - st
print('ETAPA 3')

print(f'Para {DIVISAO} nós de dados e a organização dos arquivos {FILE_ORGANIZATION[TYPE_ORGANIZATION]} temos os seguintes tempos:')
print(f'Cálculo do Hiperplano generalizado: {times["aplicate_l2"]}')
print(f'Busca no Banco de dados: {times["get_data_from_database"]}')
print(f'Treinamento do KNN: {times["fit_knn"]}')
#print(f'Busca por K vizinhos: {times["calculate_neighborhood"]}')
print(f'O número de calculos para o fit e para a busca é: {df.shape[0] + sub.shape[0]}')