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

#TODO normalisasi data numerik
print("Normalisasi Data Numerik")
Xi = arr_excel[:,1]
max = max(arr_excel[:,1])
min = min(arr_excel[:,1])

minmax = ((Xi - min) / (max - min))

print(display(minmax))
print("==============")

#TODO normalisasi data kategori
print("Normalisasi Data Kategori")
y = pd.DataFrame(Y)
y.replace({"BENIGN": 0, "DDoS": 1}, inplace=True)
# Y['label'].repl
# y.replace('BENIGN','0', inplace=True)

#print(y)
print("==============")

#TODO shuffle data
#print("Acak data untuk bootstrap")
row_len = len(arr_excel)

"""list = [1,2,3,4,5,6,7,8,9]
samp_len = round(0.5*len(list))
"""
dataset = list(arr_excel)
samp_len = round(0.5*len(arr_excel))
"""print(dataset)
acak = random.sample(dataset,samp_len)
print(acak)
print("==============")"""

print(display(arr_excel))

"""print(arr_excel)
print("==============")
print(col_len)
print(0.5*col_len)
print("==============")"""


def buildtrees(Xtrain, Ytrain):
    data = np.column_stack((Xtrain, Ytrain))
    if Xtrain.shape[0] == 1:
        return np.array([-1, Ytrain[0], 0, 0])

    f = np.arange(Xtrain.shape[1])
    e = np.arange(Xtrain.shape[0])
    numpy.random.shuffle(f)
    numpy.random.shuffle(e)

    f = f[0]
    split_val = np.mean(Xtrain[e[0:2], f])

    left_data = [data[i] for i, x in enumerate(Xtrain) if x[f] < split_val]
    right_data = [data[i] for i, x in enumerate(Xtrain) if x[f] >= split_val]

    left_dtree = buildtrees(np.array(left_data)[:, 0:Xtrain.shape[1]], np.array(left_data)[:, Xtrain.shape[1]])
    right_dtree = buildtrees(np.array(right_data)[:, 0:Xtrain.shape[1]], np.array(right_data)[:, Xtrain.shape[1]])

    if left_dtree[0].size > 1:
        node = np.array([f, split_val, 1, 1 + left_dtree.shape[0]])
    else:
        node = np.array([f, split_val, 1, 2])
    tree = np.row_stack((node, left_dtree))
    tree = np.row_stack((tree, right_dtree))
    return tree