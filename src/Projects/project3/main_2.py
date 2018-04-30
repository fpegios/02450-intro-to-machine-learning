import numpy as np
from matplotlib.pyplot import figure, plot, legend, xlabel, show, xticks, yticks
from sklearn import model_selection
import pickle
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from _load_data import *

# fetch data
gmm_f = open('gmm_data.pckl', 'rb')
gmm = pickle.load(gmm_f)
gmm_f.close()
bestK = gmm[0]
PC = gmm[1]

# Perform hierarchical/agglomerative clustering on data matrix
# Method = 'single'
# Method = 'average'
Method = 'complete'
# Metric = 'euclidean'
Metric = 'mahalanobis'

Z = linkage(PC, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = bestK
clsHIER = fcluster(Z, criterion='maxclust', t=Maxclust)

# Save data results
f = open('hier_data.pckl', 'wb')
pickle.dump([ clsHIER, Z ], f)
f.close()

# figure(1)
# clusterplot(PC, clusterid=clsHIER.reshape(clsHIER.shape[0],1), y=Y)

# Display dendrogram
max_display_levels=6
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)
xticks(fontsize=14)
yticks(fontsize=14)
show()