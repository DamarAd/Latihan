from time import asctime

import pandas as pd
import numpy as np
import random


from DecisionTree import decision_tree_algorithm, decision_tree_predictions
from helper_functions import calculate_accuracy


print("start time: ",asctime())

def bootstrapping(data, n_bootstrap):

    bootstrap_indices = np.random.randint(low=0, high=len(data), size=n_bootstrap)
    df_bootstrapped = data.iloc[bootstrap_indices]

    return df_bootstrapped

def random_forest_algorithm(data, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(data, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)

    return forest

def random_forest_predictions(data, forest):
    df_predictions = {}
    for i in range(len(forest)):
        col_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(data, tree=forest[i])
        df_predictions[col_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]

    return random_forest_predictions

col_names = ['total_length_of_fwd_packets', 'fwd_packet_length_max', 'fwd_packet_length_mean', 'avg_fwd_segment_size',
             'sublfow_fwd_bytes', 'init_win_bytes_fwd', 'act_data', 'label']

dataframe = pd.read_excel("CICIDS2017.xlsx", names=col_names)

forest = random_forest_algorithm(dataframe, n_trees=4, n_bootstrap=800, n_features=2, dt_max_depth=4)
predictions = random_forest_predictions(dataframe, forest)
accuracy = calculate_accuracy(predictions, dataframe.label)

print(forest)
print("==========")
print(predictions)
print("==========")
print(accuracy)

#print(dataframe.shape)
print("end time: ",asctime())