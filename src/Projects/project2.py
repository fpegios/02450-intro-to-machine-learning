# Project 2

import xlrd
import numpy as np
from sklearn import model_selection, tree
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot

# Load xls data file and extract variables of interest
doc = xlrd.open_workbook('dataset.xls').sheet_by_index(0)
attributeNames = doc.row_values(0, 0, 10);
numAttributes = len(attributeNames);
numInstances = doc.nrows - 1;

# Preallocate memory, then extract excel attributes data to matrix X
X = np.empty((numInstances, numAttributes))
for i in range(numAttributes):
    X[:, i] = doc.col_values(i, 1, numInstances + 1);

# Preallocate memory, then extract excel attributes data to matrix X
Y = np.empty((numInstances, 1));
Y[:, 0] = doc.col_values(numAttributes, 1, numInstances + 1);

classNames = np.unique(Y);
## Classification Problem:
# [1] In this problem we are trying to predict one class between 1,2,3,4 and 5 having known ten attributes

# [2.1] Decision Tree

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True)

# Initialize variable
Error_train = np.empty((K));
Error_test = np.empty((K));
criterion = 'gini';
max_depth = 15;

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, Y_train = X[train_index,:], Y[train_index]
    X_test, Y_test = X[test_index,:], Y[test_index]

    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    dtc = dtc.fit(X_train, Y_train.ravel())
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)

    # Evaluate misclassification rate over train/test data (in this CV fold)
    misclass_rate_test = sum(np.abs(y_est_test - Y_test)) / float(len(y_est_test))
    misclass_rate_train = sum(np.abs(y_est_train - Y_train)) / float(len(y_est_train))
    Error_test[k], Error_train[k] = np.mean(misclass_rate_test), np.mean(misclass_rate_train)
    k += 1

# print(Error_train, Error_test)
f = figure()
plot(Error_train)
plot(Error_test)
xlabel('K-Folds')
ylabel('Error (misclassification rate)')
legend(['Error_train','Error_test'])
show()