                                                                                                                                                                                                                                                                                                                                                                                                            #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 18:17:39 2023

@author: samnot
"""

import numpy as np
import h5py
from scipy.spatial import distance
import multiprocessing as mp
import threading
import csv



centers = [[ 31.65569456,  14.94642108,   8.43557493,  13.17287814,
         32.59415135,  16.19868926,   9.96176994,  15.42914739,
         74.76984826,  27.90546056,  10.7369866 ,  11.90031504,
         22.38021656,  11.50601422,   9.68891667,  27.21412282,
         57.17673623,  25.5840994 ,  14.75066895,  26.84472838,
         44.9041491 ,  15.79122095,   7.20854715,  17.18750188,
         17.59156026,   9.05603002,  10.12959027,  33.39575886,
         68.77687537,  23.93854931,   7.60404349,   8.80779656,
         40.30186464,  16.34118208,   8.87963743,  17.72530299,
         41.305617  ,  19.68880253,  10.60451809,  17.25807153,
         99.21340611,  31.2012437 ,   9.31322324,  14.0975829 ,
         28.90308095,  15.49845184,  11.20994513,  28.87650626,
         79.67042703,  22.91699098,  12.01002731,  31.40542812,
         59.00919466,  21.84301921,   7.72860298,  23.55207161,
         21.39877709,   9.03669572,   9.16722017,  40.25014148,
         89.37087292,  33.93968895,   7.75462316,   8.78766324,
         40.30720301,  17.23807699,  10.72259596,  20.11609056,
         41.42244709,  17.44855679,   8.77488565,  16.29633523,
         99.49075068,  29.17098875,  11.34805419,  15.73189875,
         28.93092444,  13.62664413,   8.65135466,  30.81051981,
         80.49608303,  23.97139714,   8.29784795,  23.49691749,
         59.13690514,  29.27021593,  10.76643053,  22.60619481,
         21.40526712,   9.02178361,   8.38217324,  35.70028428,
         89.72657541,  38.67373182,   8.40828895,   8.73783548,
         31.72522069,  15.64533082,  10.29245972,  16.67760433,
         32.78379914,  12.8744108 ,   8.15864189,  14.79080282,
         76.08883941,  28.04847845,  10.00451172,  11.70593925,
         22.22951133,  11.30334852,  10.05823782,  27.72926261,
         58.44488674,  17.6774235 ,   7.55034682,  16.4408292 ,
         44.91462818,  25.60481126,  13.90498956,  25.74260012,
         17.5448078 ,   8.81649138,   7.93006115,  25.02095576,
         69.45184115,  32.67530822,   9.64122041,   8.91466243],
       [ 26.45156199,  17.13569254,  12.11954584,  12.42164029,
         22.22700563,  27.05329583,  30.52585027,  29.97817064,
         51.89418604,  31.42134994,  18.28786558,  12.91017952,
         15.10601515,  20.75064354,  30.02013493,  42.23871152,
         33.1893035 ,  28.89140586,  25.79551387,  21.24346489,
         19.69390605,  20.34841425,  24.90900107,  27.97353701,
         21.19569464,  19.50608873,  21.29627756,  18.59208322,
         15.46513292,  16.93970304,  21.33511951,  22.49126951,
         37.93473079,  17.57030857,  12.16251379,  20.29889002,
         34.18410624,  34.10413884,  29.39033574,  28.98243119,
        113.14606161,  59.38072053,  21.54371007,  12.17684921,
         11.31768556,  12.85468582,  18.5582132 ,  44.95310691,
         44.03888058,  35.16280896,  38.24412476,  41.82542753,
         38.54006562,  27.50387734,  19.52216494,  20.38947661,
         27.04256527,  24.2459663 ,  24.32593531,  21.2695915 ,
         18.81733197,  21.55490416,  24.53206116,  26.3205887 ,
         37.9889395 ,  28.41583981,  28.84398712,  33.57787076,
         34.0475961 ,  20.60278674,  12.62466407,  18.24478335,
        113.00764556,  44.08442977,  17.84000726,  12.48822506,
         11.28887181,  12.61872614,  22.58861806,  60.31012402,
         43.09884374,  19.22152254,  18.47230562,  26.3727254 ,
         38.48909514,  43.37943393,  40.06149986,  36.31253531,
         26.93856033,  25.12874355,  23.51839752,  20.52326677,
         18.66474369,  22.45866194,  25.71758694,  25.36166071,
         26.33697167,  28.83572579,  29.42240446,  26.4067657 ,
         22.17865502,  12.82504351,  13.02050704,  18.11426296,
         50.70910446,  40.02323893,  28.71648414,  20.26466689,
         15.22803311,  13.62441492,  19.84633886,  32.94189158,
         31.99798947,  26.14209366,  23.76703681,  19.68962651,
         19.67581964,  22.40222092,  27.83513287,  30.36419721,
         21.2023227 ,  21.519875  ,  20.45376984,  16.1279929 ,
         15.03461681,  19.35915704,  22.82006579,  20.62988385],
       [ 17.54107561,  17.76112252,  18.33163748,  20.2678988 ,
         27.71544931,  23.76908464,  19.90988525,  16.4380349 ,
         29.69722804,  22.11962335,  20.34670494,  21.96424993,
         30.89513081,  26.50947318,  25.10857866,  25.76470825,
         58.38054934,  41.0917548 ,  24.85768166,  16.33016913,
         14.20678665,  13.95599351,  18.36813591,  31.95672258,
         31.66189515,  30.55130637,  27.42492816,  21.55261495,
         16.55343284,  11.28255988,  12.44472534,  18.73842639,
         22.072135  ,  21.20240345,  22.30245297,  26.73539462,
         33.94005196,  27.21741586,  22.45521691,  20.23972035,
         40.80050345,  16.28055941,  17.28521761,  31.42600686,
         51.75385115,  45.86579022,  34.62161402,  29.81299634,
        116.43915872,  47.38388874,  17.33066179,  11.3910954 ,
         11.21004886,  12.6462757 ,  19.77405018,  55.01257944,
         43.23464243,  31.15952987,  28.93233391,  30.09198666,
         27.35844776,  18.19125149,  12.43356515,  18.95150333,
         22.0783393 ,  20.5449457 ,  22.83182038,  27.43374195,
         33.94359598,  26.60437468,  22.00761703,  20.87630393,
         40.62250002,  30.15863592,  35.12233528,  46.08845371,
         51.76713951,  31.28359686,  16.94753192,  15.93183642,
        116.44500544,  55.30935966,  20.10324646,  12.73501405,
         11.22394312,  11.25836362,  16.91462489,  47.05270647,
         43.28356167,  19.23776756,  12.65506877,  18.16717811,
         27.32187979,  29.94429099,  28.50885735,  30.94298809,
         17.61843411,  16.81866767,  20.42110603,  24.02125247,
         27.70412565,  20.07161821,  17.84971018,  17.39842296,
         29.65033395,  26.31617582,  25.77479627,  26.81770533,
         30.88890607,  21.71140383,  19.7148979 ,  21.54022536,
         58.33775437,  32.51376074,  18.93640635,  14.180515  ,
         14.2654542 ,  16.03479936,  23.98751449,  40.40244261,
         31.6738931 ,  19.10401724,  12.85672215,  11.4043883 ,
         16.55423087,  21.28018367,  26.70919535,  30.13896901],
       [ 50.36853228,  20.62865392,   6.25626805,   5.830525  ,
         14.03908541,  10.30411507,  10.40038686,  22.26749709,
         95.55886873,  32.66884464,   6.83345483,   3.92232259,
          6.19934441,   5.12772842,   8.59920336,  35.09735235,
         97.02215422,  37.40648103,   9.134646  ,   4.69322481,
          4.94456776,   3.54775503,   5.91312332,  31.09228962,
         53.25008642,  25.03042381,  11.17257   ,   8.81543436,
          7.79786863,   4.26591788,   5.3198528 ,  19.27328531,
         63.176241  ,  24.01754622,   7.25855288,   8.02186047,
         16.58725267,   9.85470794,   9.17768744,  24.72192514,
        115.51883307,  36.34778653,   6.84468387,   5.11241546,
          7.99245207,   6.24188408,   9.51294972,  35.66655005,
        118.14126808,  36.24676632,   8.55341851,   5.57858723,
          6.49026271,   4.43089151,   6.27973351,  37.38284029,
         67.38398009,  26.57221604,   9.24242837,   7.94548699,
          8.86012958,   5.90317974,   6.39898501,  23.5673101 ,
         63.10287645,  24.93712262,   9.26582728,   9.94578492,
         16.62810127,   8.02126949,   7.19370359,  23.69012946,
        115.6658845 ,  36.02445717,   9.60321968,   6.2267349 ,
          8.00349119,   5.0892204 ,   6.54179745,  36.0146359 ,
        118.67020188,  38.01429337,   6.6488756 ,   4.59212201,
          6.45675749,   5.28910144,   7.80413249,  35.64848219,
         67.44175013,  24.33035007,   6.81834433,   6.01453276,
          8.86885496,   7.81309621,   8.70600901,  25.79612948,
         50.37270715,  22.83817085,  10.73276983,  10.42206648,
         14.04144516,   5.76079242,   6.01639143,  20.11712347,
         96.14094436,  36.2588996 ,   8.97916156,   5.18320666,
          6.1780467 ,   3.78322471,   6.34981002,  31.84519261,
         97.68201842,  32.102244  ,   6.26096282,   3.60916331,
          4.89730266,   4.48391765,   8.46953264,  36.58920333,
         53.26770581,  19.8892979 ,   5.63174857,   4.37551397,
          7.86317754,   8.73911071,  10.68054897,  24.37033032]]

f = h5py.File('./SIFT10Mfeatures.mat','r')
data = f.get('fea')
data = np.array(data)

dict_set = {
    'p1':[],
    'p2':[],
    'p3':[],
    'p4':[],
}

half = int(data.shape[0]/2)

def cal_dist_and_add_to_set(point:list, index:int):
    global dict_set
    dist_list = []
    for i in range(4):
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



#print(f'P1:{len(dict_set["p1"])}, P2:{len(dict_set["p2"])}, P3:{len(dict_set["p3"])}, P4:{len(dict_set["p4"])}')
#with open ("./novo.json","a") as arq:
#    json.dump(arq, dict_set)
#

print(dict_set['p1'])
header = ['gh']+[f'fea{i}' for i in range(128)]
with open(f'./inserts_full.csv', mode='w') as arq:
    print('foi aqui')
    arq = csv.writer(arq, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    arq.writerow(header)
    for chave, valor in dict_set.items():        
        for item in valor:
            arq.writerow([chave[-1]]+[str(char_number) for char_number in data[item]])
        
        

# print(set_p1)
# def cal_dist_and_add_to_set(point:list):
#     global set_p1, set_p2
#     d_p1 = distance.euclidean(point, data[p1])
#     d_p2 = distance.euclidean(point, data[p2])
#     print(f'P1:{d_p1}, P2:{d_p2}')
#     if d_p1 != 0 and d_p2 != 0:
#         if d_p1 < d_p2 :
#             set_p1.append(point)
#             return 0
#         else:
#             set_p2.append(point)
#             return 1
            
  
# with mp.Pool(4) as p:
#     p.map(cal_dist_and_add_to_set, data)

