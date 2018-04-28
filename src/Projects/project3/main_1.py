import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
import pickle
from scipy.linalg import svd
from toolbox_02450 import clusterplot
from _load_data import *

# Range of K's to try
KRange = range(1,11)
T = len(KRange)

covar_type = 'full'     # you can try out 'diag' as well
reps = 3                # number of fits with different initalizations, best result will be kept

# Allocate variables
CVE = np.zeros((T,))
minCV = float('Inf')
bestK = 0;

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10, shuffle=True)

for t, K in enumerate(KRange):
    print('Fitting model for K={0}'.format(K))

    # For each crossvalidation fold
    for train_index, test_index in CV.split(X):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        X_test = X[test_index]

        # Fit Gaussian mixture model to X_train
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

        # compute negative log likelihood of X_test
        CVE[t] += -gmm.score_samples(X_test).sum()

    if CVE[t] < minCV:
        bestK = K
        minCV = CVE[t]

## PCA
Xst = X - np.ones((N,1))*X.mean(0)
# PCA by computing SVD of Y
U,S,V = svd(Xst, full_matrices=False)
V = V.T
# Project data onto principal component space
Z = Xst @ V
# Indices of the principal components to be plotted
i = 0
j = 1
# Principal Components
PC = np.zeros((numInstances, 2))

for c in range(C):
    # select indices belonging to class c:
    class_mask = Y==c
    PC[class_mask, i] = Z[class_mask,i]
    PC[class_mask, j] = Z[class_mask,j]

# Fit Gaussian mixture model to Principal Components
gmm = GaussianMixture(n_components=bestK, covariance_type=covar_type, n_init=reps).fit(PC)
# extract cluster labels
clsGMM = gmm.predict(PC)
# extract cluster centroids (means of gaussians)
cdsGMM = gmm.means_
# extract cluster shapes (covariances of gaussians)
covsGMM = gmm.covariances_

# Save data results
f = open('gmm_data.pckl', 'wb')
pickle.dump([ bestK, PC, gmm, clsGMM, cdsGMM, covsGMM, CVE, KRange ], f)
f.close()

