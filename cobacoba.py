#from RandomForest import *
import numpy as np
import pandas as pd
from pprint import pprint

from sklearn.model_selection import train_test_split

from helper_functions import determine_type_of_feature
#from DecisionTree import *
from RandomForest import random_forest_algorithm, random_forest_predictions
col_names = ['total_length_of_fwd_packets', 'fwd_packet_length_max', 'fwd_packet_length_mean', 'avg_fwd_segment_size',
             'sublfow_fwd_bytes', 'init_win_bytes_fwd', 'act_data', 'label']

secondary = pd.read_excel("ddos_cicids2017.xlsx", names=col_names)
train_sekunder, test_sekunder = train_test_split(secondary, train_size= 0.25)

primary = pd.read_excel("data_primer.xlsx", names=col_names)
#primer = pd.DataFrame(primary.values)
train_primer, test_primer = train_test_split(primary, train_size= 0.5)

#print(secondary.info())
#print("===========")
#print(primary.info())
"""
tree = decision_tree_algorithm(train_primer, max_depth=4)
pprint(tree)
print("=======")
"""
forest = random_forest_algorithm(train_sekunder, n_trees=10, n_bootstrap=200, n_features=6, dt_max_depth=8)
prediction = random_forest_predictions(test_sekunder, forest)
unique_clases, counts_unique_clasess = np.unique(prediction, return_counts=True)


print(prediction)
pprint(test_sekunder.label)
print(unique_clases)
print(counts_unique_clasess)

