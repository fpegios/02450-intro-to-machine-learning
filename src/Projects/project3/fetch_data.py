import pickle
import numpy as np
from matplotlib.pyplot import figure, plot, ylim, title, legend, xlabel, ylabel, show
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import dendrogram
from _load_data import *


########################################################
# FETCH DATA
gmm_f = open('gmm_data.pckl', 'rb')
gmm = pickle.load(gmm_f)
gmm_f.close()
bestK = gmm[0]
PC = gmm[1]
GMM = gmm[2]
clsGMM = gmm[3]
cdsGMM = gmm[4]
covsGMM = gmm[5]
CVE = gmm[6]
KRange = gmm[7]

hier_f = open('hier_data.pckl', 'rb')
hier = pickle.load(hier_f)
hier_f.close()
clsHIER = hier[0]
Z = hier[1]

eval_f = open('eval_data.pckl', 'rb')
eval = pickle.load(eval_f)
eval_f.close()
rand = eval[0]
jaccard = eval[1]
nmi = eval[2]
########################################################

########################################################
# PART 1 - GMM CLUSTERING
print('==============================================')
print('Best K: {0}'.format(bestK))
print('==============================================')

# Cluster Plot
figure
clusterplot(PC, clusterid=clsGMM, centroids=cdsGMM, covars=covsGMM, y=Y)
show()

# Plot CV error per K
figure
plot(KRange, 2*CVE,'-ok')
legend(['Crossvalidation Error'])
xlabel('K')
show()
########################################################


########################################################
# PART 2 - HIERARCHICAL CLUSTERING
figure(1)
clusterplot(PC, clusterid=clsHIER.reshape(clsHIER.shape[0],1), y=Y)

# Display dendrogram
max_display_levels=6
figure(2, figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()
########################################################


########################################################
# PART 3 - QUALITY EVALUATION
print('\n')
print('==============================================')
print('=== GMM ===')
print('Rand: {0}'.format(rand[0]))
print('Jaccard: {0}'.format(jaccard[0]))
print('NMI: {0}'.format(nmi[0]))
print('==============================================')

print('\n')
print('==============================================')
print('=== HIERARCHICAL ===')
print('Rand: {0}'.format(rand[1]))
print('Jaccard: {0}'.format(jaccard[1]))
print('NMI: {0}'.format(nmi[1]))
print('==============================================')
########################################################