create database if not exists dataset_full;
use dataset_full;
create external table if not exists dataset_full_table (gh int,fea0 int,fea1 int,fea2 int,fea3 int,fea4 int,fea5 int,fea6 int,fea7 int,fea8 int,fea9 int,fea10 int,fea11 int,fea12 int,fea13 int,fea14 int,fea15 int,fea16 int,fea17 int,fea18 int,fea19 int,fea20 int,fea21 int,fea22 int,fea23 int,fea24 int,fea25 int,fea26 int,fea27 int,fea28 int,fea29 int,fea30 int,fea31 int,fea32 int,fea33 int,fea34 int,fea35 int,fea36 int,fea37 int,fea38 int,fea39 int,fea40 int,fea41 int,fea42 int,fea43 int,fea44 int,fea45 int,fea46 int,fea47 int,fea48 int,fea49 int,fea50 int,fea51 int,fea52 int,fea53 int,fea54 int,fea55 int,fea56 int,fea57 int,fea58 int,fea59 int,fea60 int,fea61 int,fea62 int,fea63 int,fea64 int,fea65 int,fea66 int,fea67 int,fea68 int,fea69 int,fea70 int,fea71 int,fea72 int,fea73 int,fea74 int,fea75 int,fea76 int,fea77 int,fea78 int,fea79 int,fea80 int,fea81 int,fea82 int,fea83 int,fea84 int,fea85 int,fea86 int,fea87 int,fea88 int,fea89 int,fea90 int,fea91 int,fea92 int,fea93 int,fea94 int,fea95 int,fea96 int,fea97 int,fea98 int,fea99 int,fea100 int,fea101 int,fea102 int,fea103 int,fea104 int,fea105 int,fea106 int,fea107 int,fea108 int,fea109 int,fea110 int,fea111 int,fea112 int,fea113 int,fea114 int,fea115 int,fea116 int,fea117 int,fea118 int,fea119 int,fea120 int,fea121 int,fea122 int,fea123 int,fea124 int,fea125 int,fea126 int,fea127 int)
row format delimited
fields terminated by ','
lines terminated by '\n'
LOCATION 'hdfs://namenode:8020/user/hive/warehouse/dataset_full.db/dataset_full_table';

