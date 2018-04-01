import pickle
from _load_data import *

f = open('dec_tree_data.pckl', 'rb')
a = pickle.load(f)
f.close()

# print(a[0])

y = a[0].predict(X);

print(y)