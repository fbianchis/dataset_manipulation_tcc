#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:44:08 2023

@author: samnot
"""
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import h5py
import json
import numpy
from json import JSONEncoder
import threading
import csv
from scipy.spatial import distance

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


f = h5py.File('./SIFT10Mfeatures.mat','r')
data = f.get('fea')
data = np.array(data)
full = True
n_clusters = 1

#kmeans = KMeans(n_clusters=256, random_state=0, n_init="auto").fit(data[:100000][::]) 

batch_size = 1024

kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                          random_state=0,
                          batch_size=batch_size,
                          n_init="auto")

step = int(data.shape[0]/batch_size)

for i in range(batch_size):
    kmeans = kmeans.partial_fit(data[i*step:(i+1)*(step),:])
    #print(f'INI:{i*step}, FIM:{(i+1)*(step)}')

if (i+1)*(step) < data.shape[0]:
    #print(f'TAMANHO ULTIMO INT: {len(data[(i+1)*(step)::])}')
    kmeans = kmeans.partial_fit(data[(i+1)*(step)::,:])

print(f'TOTAL {data.shape[0]}')
numpyData = {"array": kmeans.cluster_centers_}
centers = kmeans.cluster_centers_
np.save(f'centers_{n_clusters}_kmeans.npy', centers)

dict_set = {
    'p1':[],
    #'p2':[],
    # 'p3':[],
    # 'p4':[],
    # 'p5':[],
    # 'p6':[],
    # 'p7':[],
    # 'p8':[],
}

half = int(data.shape[0]/2)

def cal_dist_and_add_to_set(point:list, index:int):
    global dict_set
    dist_list = []
    for i in range(n_clusters):
        dist_list.append(distance.euclidean(point, centers[i]))
    
    index_novo = dist_list.index(min(dist_list))
    dict_set[f'p{index_novo+1}'].append(index)
    print(index)

def thread_func():
    global dict_set
    for i in range(0,half,1):
        point = data[i]
        cal_dist_and_add_to_set(point, i)
    print('FIM THREAD')


x = threading.Thread(target=thread_func)
x.start()

for j in range(half +1 ,data.shape[0],1):
    point = data[j]
    cal_dist_and_add_to_set(point, j)


print(dict_set['p1'])
header = ['gh']+[f'fea{i}' for i in range(128)]

if full:
    with open(f'inserts_full_{n_clusters}_clusters.csv', mode='w') as arq:
        arq = csv.writer(arq, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        arq.writerow(header)
        for chave, valor in dict_set.items():    
                for item in valor:
                    arq.writerow([chave[-1]]+[str(char_number) for char_number in data[item]])

else:
    for chave, valor in dict_set.items():
        with open(f'inserts_{chave}_of_{n_clusters}.csv', mode='w') as arq:
            arq = csv.writer(arq, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            arq.writerow(header)
            for item in valor:
                arq.writerow([chave[-1]]+[str(char_number) for char_number in data[item]])