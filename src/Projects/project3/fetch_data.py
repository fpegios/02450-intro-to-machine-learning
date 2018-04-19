import pickle
import numpy as np
from matplotlib.pyplot import figure, plot, legend, xlabel, show
from toolbox_02450 import clusterplot
from _load_data import *

# fetched saved model data
gmm_f = open('gmm_data.pckl', 'rb')
gmm = pickle.load(gmm_f)
gmm_f.close()

# Stored saved data to variables
bestK = gmm[0]
best_gmm = gmm[1]
cls = gmm[2]
cds = gmm[3]
CVE = gmm[4]
KRange = gmm[5]

# Plot CV error per K
figure(1);
plot(KRange, 2*CVE,'-ok')
legend(['Crossvalidation'])
xlabel('K')


print('==============================================')
print('Best K: {0}'.format(bestK))
print('==============================================')

# Cluster Plot
figure(figsize=(14,9))
clusterplot(X, clusterid=cls, centroids=cds, y=Y)
show()