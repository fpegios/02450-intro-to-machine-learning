# LOAD DATA ###################################################
import xlrd
import numpy as np
from scipy import stats

doc = xlrd.open_workbook('../../dataset.xls').sheet_by_index(0)
numAttributes = 10
attributeNames = doc.row_values(0, 0, numAttributes)
numInstances = doc.nrows - 1

# Preallocate memory, then extract attributes data to matrix X
X = np.empty((numInstances, numAttributes))
for i in range(numAttributes):
    X[:, i] = doc.col_values(i, 1, numInstances + 1)

# Preallocate memory, then extract output data to matrix Y
Y = np.empty((numInstances, 1));
Y[:, 0] = doc.col_values(numAttributes, 1, numInstances + 1)

# Normalize data
X = stats.zscore(X)

# save output class names
classNames = np.unique(Y)
C = len(classNames)
N, M = X.shape

# X = X[0:100,:]