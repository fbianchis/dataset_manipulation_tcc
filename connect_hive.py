from pyhive import hive

import pandas as pd

#Create Hive connection 

conn = hive.Connection(host="192.168.1.24", port=10000, username="username")

# Read Hive table and Create pandas dataframe

df = pd.read_sql("SELECT * FROM dataset_partition_4.dataset_partition_table where gh == 1", conn)

print(df.loc[0][:])

conn.close()