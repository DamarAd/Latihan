from array import *
import pandas as pd
T = [[11, 12, 5, 2], [15, 6,10], [10, 8, 12, 5], [12,15,8,6]]

def display(arr):
    return pd.DataFrame(arr)

"""
for r in T:
    for c in r:
        print(c,end = " ")
    print()
"""

for r in T:
    for c in r:
        print(c, end=" ")
    print()
print("=======")

t2d = display(T)
print([b[0] for b in T])