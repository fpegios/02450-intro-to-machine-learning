# LOAD DATA ###################################################
import xlrd
import numpy as np
from scipy import stats
from matplotlib.pyplot import figure, title, subplot, plot, boxplot, xlabel, ylabel, show, hist

doc = xlrd.open_workbook('../../dataset.xls').sheet_by_index(0)
numAttributes = 10
attributeNames = doc.row_values(0, 0, numAttributes)
attributeNames = np.append(['one'], attributeNames)
numInstances = doc.nrows - 1

print(attributeNames)

# Preallocate memory, then extract attributes data to matrix X
X = np.ones((numInstances, numAttributes + 1))

for i in range(numAttributes):
    X[:, i + 1] = doc.col_values(i, 1, numInstances + 1)

# Preallocate memory, then extract output data to matrix Y
Y = np.empty((numInstances, 1));
Y[:, 0] = doc.col_values(numAttributes, 1, numInstances + 1)

# # Normalize data
# X = stats.zscore(X)
#
# # save output class names
classNames = np.unique(Y)
C = len(classNames)
N, M = X.shape

# X = X[0:150,:]

# data = X[:, 0];

# figure()
#
# subplot(1,3,1)
# hist(X[:,1])
# title('height')
# subplot(1,3,2)
# hist(X[:,2])
# title('length')
# subplot(1,3,3)
# hist(X[:,4])
# title('p_and')
#
# show()

# correlation = np.empty((numAttributes, numAttributes))
#
# for i in range(numAttributes):
#     for j in range(numAttributes):
#         correlation[i, j] = np.corrcoef(X[:, i], X[:, j])[0, 1]
#
# print(correlation)