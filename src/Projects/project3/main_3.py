import numpy as np
from matplotlib.pyplot import figure, plot, ylim, title, legend, xlabel, ylabel, show
import pickle
from toolbox_02450 import clusterval
from _load_data import *

# fetch data
gmm_f = open('gmm_data.pckl', 'rb')
gmm = pickle.load(gmm_f)
gmm_f.close()
bestK = gmm[0]
clsGMM = gmm[3]

hier_f = open('hier_data.pckl', 'rb')
hier = pickle.load(hier_f)
hier_f.close()
clsHIER = hier[0]

# Quality Evaluation
# Allocate variables:
Rand = np.zeros((2))
Jaccard = np.zeros((2))
NMI = np.zeros((2))

# compute cluster validities:
Rand[0], Jaccard[0], NMI[0] = clusterval(Y, clsGMM)
Rand[1], Jaccard[1], NMI[1] = clusterval(Y, clsHIER)

# Save data results
f = open('eval_data.pckl', 'wb')
pickle.dump([ Rand, Jaccard, NMI], f)
f.close()