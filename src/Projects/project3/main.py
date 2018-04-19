import numpy as np
from matplotlib.pyplot import figure, plot, legend, xlabel, show
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
import pickle
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
        print(K)
        bestK = K
        minCV = CVE[t]

# Fit Gaussian mixture model to X_train
best_gmm = GaussianMixture(n_components=bestK, covariance_type=covar_type, n_init=reps).fit(X)
# extract cluster labels
cls = best_gmm.predict(X)
# extract cluster centroids (means of gaussians)
cds = best_gmm.means_

# Save results
f = open('gmm_data.pckl', 'wb')
pickle.dump([bestK,
             best_gmm,
             cls,
             cds,
             CVE,
             KRange
             ],
            f)
f.close()

