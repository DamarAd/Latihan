from time import *
from pprint import pprint

import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split

from DecisionTree import decision_tree_algorithm, decision_tree_predictions
from helper_functions import localtime_in_sec

# Option Display for Rows n Columns
desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 100)


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

dataframe = pd.read_excel("ddos_cicids2017.xlsx", names=col_names)
primary = pd.read_excel("data_primer.xlsx", names=col_names)

secondary, test_df = train_test_split(dataframe, train_size= 0.01)

forest = random_forest_algorithm(secondary, n_trees=50, n_bootstrap=800, n_features=6, dt_max_depth=10)

pprint(forest)
print("==========")

now = localtime_in_sec(localtime)
print("Starting time:", now, "second")
predictions = random_forest_predictions(secondary, forest)

print(predictions)
print(secondary.label)
print("==========")

later = localtime_in_sec(localtime)
print("Ending time: ",later, "second")
duration = int(later-now)

print("==========")
from sklearn.metrics import confusion_matrix, classification_report

tn, fp, fn, tp = confusion_matrix(secondary.label, predictions).ravel()
print(" True Negative: ",tn,"\n",
      "False Positive: ",fp,"\n",
      "False Negative: ",fn,"\n",
      "True Positive: ",tp)

if (tp + tn + fp + fn) != 0:
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
else:
    accuracy = 0

if (tp+fp) != 0:
    precision = (tp / (tp + fp)) * 100
else:
    precision = 0

if (tp+fn) != 0:
    recall = (tp / (tp + fn)) * 100
else:
    recall = 0

if (precision+recall) != 0:
    f_measure = 2 * ((precision * recall) / (precision + recall))
else:
    f_measure = 0


print("duration", duration, "second")
print("akurasi: ", accuracy)
print("presisi: ", precision)
print("recall: ",recall)
print("f-measure: ",f_measure)

"""
n_bootstrap = 400
bootstrapped = bootstrapping(primary, n_bootstrap)
print(bootstrapped)
"""

#print(classification_report(primary.label, predictions))

#print(confusion_matrix(train_df.label, predictions))


