###############################################################
# CLASSIFICATION PROBLEM
###############################################################
import numpy as np
from sklearn import model_selection, tree

from _load_data import *

# [1] ##########################################################################
# In this problem we are trying to predict one class between
# 1,2,3,4 and 5 having known 10 attributes

# Initialize variables
K_out = 5
K_in = 10

# [2.1] DECISION TREE ##########################################################
max_depth = 20
criterion = 'gini'
tree_complexity = np.arange(2, max_depth + 1, 1)  # Tree complexity parameter - constraint on maximum depth
error_per_depth = np.empty((max_depth - 1, K_in))  # error per depth for each k_in
avg_error_depth = np.empty(max_depth - 1)  # average error per depth
best_depth = np.empty(K_out)    # best error per k_out
error_dec_tree = np.empty(K_out)  # error per model

# Outer K-fold crossvalidation
CV_out = model_selection.KFold(n_splits=K_out, shuffle=True)

k_out = 0
print('===========================')
print('     DECISION TREE')
print('===========================')
for train_out_index, test_out_index in CV_out.split(X):
    print('\nComputing CV_out fold: {0}/{1}..'.format(k_out + 1, K_out))

    # extract training and test set for current CV_out fold
    X_train_out, Y_train_out = X[train_out_index,:], Y[train_out_index]
    X_test_out, Y_test_out = X[test_out_index,:], Y[test_out_index]

    # Inner K-fold crossvalidation
    CV_in = model_selection.KFold(n_splits=K_in, shuffle=True)

    k_in = 0
    for train_in_index, test_in_index in CV_in.split(X_train_out):
        print('    Computing CV_in fold: {0}/{1}..'.format(k_in + 1, K_in))

        # extract training and test set for current CV_in fold
        X_train_in, Y_train_in = X[train_in_index, :], Y[train_in_index]
        X_test_in, Y_test_in = X[test_in_index, :], Y[test_in_index]

        for depth_index, depth in enumerate(tree_complexity):
            # Fit decision tree classifier with different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion=criterion, max_depth=depth)
            dtc = dtc.fit(X_train_in, Y_train_in.ravel())
            y_est_test_in = dtc.predict(X_test_in)

            # find estimated test error for each X_test_in value
            error_y_est_test_in = sum(np.abs(y_est_test_in - Y_test_in)) / float(len(y_est_test_in))
            # find average estimated test error per depth for each k_out
            error_per_depth[depth_index, k_out] = error_y_est_test_in.mean()
        k_in += 1

    # we find the mean error per depth, and we take the depth with the minimum error
    minError = 1
    for depth_index, depth in enumerate(tree_complexity):
        avg_error_depth[depth_index] = error_per_depth[depth_index, :].mean()
        if avg_error_depth[depth_index] < minError:
            minError = avg_error_depth[depth_index]
            best_depth[k_out] = depth

    # we create a model using the selected depth and find its error
    dtc = tree.DecisionTreeClassifier(criterion=criterion, max_depth=best_depth[k_out])
    dtc = dtc.fit(X_train_out, Y_train_out.ravel())
    y_dec_tree = dtc.predict(X_test_out)

    # we calculate the final error of each model
    error_dec_tree[k_out] = (sum(np.abs(y_dec_tree - Y_test_out)) / float(len(y_dec_tree))).mean();

    # Display results
    print('    =============================')
    print('    Best Depth: {0}'.format(best_depth[k_out]))
    print('    Error: {0}'.format(error_dec_tree[k_out]))
    print('\n')
    k_out += 1

# knowing the error for the 5 folds, we find the mean of the errors and this is the final error of the method
print('=======================================')
print('Decision Tree Error: {0}'.format(error_dec_tree.mean()))
print('=======================================')
print('\n')

# [2.2] ##########################################################

# [2.3] ##########################################################

# [3] ############################################################

# [4] ############################################################

