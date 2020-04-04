import numpy as np
import pandas as pd

#Option Display for Rows n Columns
desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

#Display array to dataframe object
def display(arr):
    return pd.DataFrame(arr)


#TODO Initialize of Dataset
#Import dataset
col_names = ['total_length_of_fwd_packets', 'total_length_of_bwd_packets',
    'fwd_packet_length_max', 'fwd_packet_length_min','fwd_packet_length_mean','label']

data = pd.read_excel("excel_latihan.xlsx", names=col_names)

row_len = len(data.values)
col_len = len(data.values[0])

X = data.values[:,:-1]
Y = data.values[:,-1]

"""
print(display(data.values))
print("===========")
print(display(X))
"""

# TODO Preprocess
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

scaler = MinMaxScaler()
fit_X = scaler.fit(X)
Rescaled_X = scaler.transform(X)

encoder = LabelEncoder()
fit_Y = encoder.fit(Y)
Rescaled_Y = encoder.transform(Y)

print(display(Y))
print("===========")
print(display(Rescaled_Y))