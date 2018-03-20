# Project 2

import xlrd
import numpy as np
from sklearn import model_selection, tree
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot

###############################################################
# CLASSIFICATION PROBLEM
###############################################################

# LOAD DATA
###############################################################
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
###############################################################


# [1] In this problem we are trying to predict one class between
# 1,2,3,4 and 5 having known ten attributes

# [2.1] Decision Tree

K_out = 5
K_in = 10;

# Outer K-fold crossvalidation
CV_out = model_selection.KFold(n_splits=K_out, shuffle=True)

# Initialize variable
Error_train = np.empty((K_out));
Error_test = np.empty((K_out));
criterion = 'gini';
max_depth = 15;
error_depths = np.arra
k_out = 0
for train_out_index, test_out_index in CV_out.split(X):
    print('Computing CV_out fold: {0}/{1}..'.format(k_out + 1, K_out))
    # extract training and test set for current CV_out fold
    X_train_out, Y_train_out = X[train_out_index,:], Y[train_out_index]
    X_test_out, Y_test_out = X[test_out_index,:], Y[test_out_index]
    # Inner K-fold crossvalidation
    CV_in = model_selection.KFold(n_splits=K_in, shuffle=True)

    k_in = 0;
    for train_in_index, test_in_index in CV_in.split(X_train_out):
        print('Computing CV_in fold: {0}/{1}..'.format(k_in + 1, K_in))
        # extract training and test set for current CV_in fold
        X_train_in, Y_train_in = X[train_in_index, :], Y[train_in_index]
        X_test_in, Y_test_in = X[test_in_index, :], Y[test_in_index]

        for depth_index in range(max_depth):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion=criterion, max_depth=depth_index)
            dtc = dtc.fit(X_train_in, Y_train_in.ravel())
            y_est_test = dtc.predict(X_test_in)

            test_error = sum(np.abs(y_est_test - Y_test_in)) / float(len(y_est_test))
            # create a 2d matrix [20 = depth, 10 = fold] that contains the error

        k_in += 1;

         # we find the mean error per depth, and we take the depth with the minimum error
         # then we create a model using the selected depth and find its erro
         # and we found a final error of the 5 models calculating the mean error of the five ones

    # knowing the error for the 5 folds, we find the mean of the errors and this is the final error of the method

    # 2.2
    # find the weight of all the methods' models

    # 2.3
    # show a boxplot of the 5 errors in each method model
    # find the mode of the output of the train data
    # find the misclassification rate over train-mode/test-output data (numpy mode)

    # Evaluate misclassification rate over train/test data (in this CV fold)
    misclass_rate_test = sum(np.abs(y_est_test - Y_test)) / float(len(y_est_test))
    misclass_rate_train = sum(np.abs(y_est_train - Y_train)) / float(len(y_est_train))
    Error_test[k], Error_train[k] = np.mean(misclass_rate_test), np.mean(misclass_rate_train)
    k_out += 1

# print(Error_train, Error_test)
f = figure()
plot(Error_train)
plot(Error_test)
xlabel('K-Folds')
ylabel('Error (misclassification rate)')
legend(['Error_train','Error_test'])
show()