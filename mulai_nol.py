import numpy
import numpy as np
import pandas as pd
import random

# Option Display for Rows n Columns
desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# Display array to dataframe
def display(arr):
    return pd.DataFrame(arr)


col_names= ['total_length_of_fwd_packets', 'total_length_of_bwd_packets',
    'fwd_packet_length_max', 'fwd_packet_length_min','fwd_packet_length_mean','label']

print()
excel = pd.read_excel("excel_latihan.xlsx")
arr_excel = excel.values

row_len = len(arr_excel)
col_len = len(arr_excel[0])

print(display(arr_excel))
print("===============================")

X = arr_excel[:,:-1]
Y = arr_excel[:,-1]


# TODO Preprocess
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
fit_X = scaler.fit(arr_excel[:, :-1])
Rescaled_dataX = scaler.transform(arr_excel[:, :-1])
print(Rescaled_dataX)

datalengkap = np.vstack([Rescaled_dataX,Y])
print(datalengkap)

#TODO normalisasi data numerik
print("Normalisasi Data Numerik")
col = 0+1
Xi = arr_excel[:,col]
max = max(arr_excel[:,col])
min = min(arr_excel[:,col])

minmax = ((Xi - min) / (max - min))

print(display(minmax))
print("==============")

_,n_col = arr_excel.shape

for x in range(n_col-2):
    x+=1


    xi = arr_excel[:, x]
    print(xi)



"""
Xi = arr_excel[:, 0]
max = max(arr_excel[:, 0])
min = min(arr_excel[:, 0])

minmax = ((Xi - min) / (max - min))
"""

