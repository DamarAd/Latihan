import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# Option Display for Rows n Columns

desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# Display array to dataframe object
def display(arr):
    return pd.DataFrame(arr)


# TODO Initialize of Dataset
# Import dataset

col_names = ['total_length_of_fwd_packets', 'total_length_of_bwd_packets',
             'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean', 'label']

dataframe = pd.read_excel("excel_latihan.xlsx", names=col_names)

"""
col_names = ['total_length_of_fwd_packets', 'fwd_packet_length_max', 'fwd_packet_length_mean', 'avg_fwd_segment_size',
             'sublfow_fwd_bytes', 'init_win_bytes_fwd', 'act_data', 'label']

dataframe = pd.read_excel("ddos_cicids2017.xlsx", names=col_names)
"""
train_df, test_df = train_test_split(dataframe, test_size = 0.2)

row_len = len(dataframe.values)
col_len = len(dataframe.values[0])

X = dataframe.values[:, :-1]
Y = dataframe.values[:, -1]

"""
print(display(data.values))
print("===========")
print(display(X))
"""

# TODO Preprocess
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

scaler = MinMaxScaler()
fit_X = scaler.fit(dataframe.values[:, :-1])
Rescaled_dataX = scaler.transform(dataframe.values[:, :-1])

encoder = LabelEncoder()
fit_Y = encoder.fit(dataframe.values[:, -1])
Rescaled_dataY = encoder.transform(dataframe.values[:, -1])


# TODO Build Tree
# Check purity
def check_purity(data):
    label = data[:, -1]
    label_column = encoder.transform(label)
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


# Classify
def clasify_data(data):
    label = data[:, -1]
    label_column = encoder.transform(label)
    unique_clases, counts_unique_clasess = np.unique(label, return_counts=True)

    index = counts_unique_clasess.argmax()
    classification = unique_clases[index]

    return classification


# Potential Splits
def get_potential_splits(data):
    potential_splits = {}
    _, n_col = data.shape
    for col_index in range(n_col - 1):
        potential_splits[col_index] = []
        values = data[:, col_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2

                potential_splits[col_index].append(potential_split)

    return potential_splits


# Split Data
def split_data(data, split_col, split_value):
    split_col_values = data[:, split_col]

    data_below = data[split_col_values <= split_value]
    data_above = data[split_col_values > split_value]

    return data_below, data_above


# Calculate impurity
def calculate_impurity(data):
    label = data[:, -1]
    label_col = encoder.transform(label)
    _, counts = np.unique(label, return_counts=True)

    probabilities = counts / counts.sum()
    impurity = 1 - (sum(probabilities ** 2))

    return impurity


# Calculate gini index
def calculate_gini(data_below, data_above):
    ni_data_below = len(data_below)
    ni_data_above = len(data_above)
    n_total = len(data_below) + len(data_above)

    p_data_below = ni_data_below / n_total
    p_data_above = ni_data_above / n_total

    gini = (p_data_below * calculate_impurity(data_below) + p_data_above * calculate_impurity(data_above))

    return gini


# Get lowest gini value
def determine_lowest_gini(data, potential_splits):
    gini = 999
    for col_index in potential_splits:
        for value in potential_splits[col_index]:
            data_below, data_above = split_data(data, split_col=col_index, split_value=value)
            current_gini = calculate_gini(data_below, data_above)

            if current_gini <= gini:
                gini = current_gini
                best_split_col = col_index
                best_split_value = value
                lowest_gini = gini

    return best_split_col, best_split_value, lowest_gini

split_col = 0
split_value = 18

data_below, data_above = split_data(dataframe.values, split_col, split_value)
"""
print(data_below)
print("=======")
print(data_above)
"""
potential_splits = get_potential_splits(dataframe.values)

print("=======")
print(determine_lowest_gini(dataframe.values, potential_splits))
