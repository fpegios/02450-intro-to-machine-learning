###############################################################
# CLASSIFICATION PROBLEM
###############################################################
import numpy as np
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier
import graphviz
import pickle

from _load_data import *

# Initialize variables
K_out = 5
K_in = 10

def dec_tree():
    max_depth = 20
    criterion = 'gini'
    tree_complexity = np.arange(2, max_depth + 1, 1)  # Tree complexity parameter - constraint on maximum depth
    error_per_depth = np.empty((max_depth - 1, K_in))  # error per depth for each k_in
    avg_error_depth = np.empty(max_depth - 1)  # average error per depth
    best_depth = np.empty(K_out)    # best error per k_out
    error_dec_tree = np.empty(K_out)  # error per model

    best_dec_tree_model_error = 1;

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

        # save the model with the lowest error
        if error_dec_tree[k_out] <= best_dec_tree_model_error:
            best_dec_tree_model_error = error_dec_tree[k_out]
            best_dec_tree_model = dtc

        # Display results
        print('    =============================')
        print('    Best Depth: {0}'.format(best_depth[k_out]))
        print('    Error: {0}'.format(error_dec_tree[k_out]))
        print('\n')
        k_out += 1

    # knowing the error for the 5 folds, we find the mean of the errors and this is the final error of the method
    print('==============================================')
    print('Best Model Depth = {0}'.format(best_dec_tree_model.max_depth))
    print('Best Model Error: {0}'.format(best_dec_tree_model_error))
    print('==============================================')
    print('Decision Tree Error: {0}'.format(error_dec_tree.mean()))
    print('==============================================')
    print('\n')

    # save the best tree, the best model error,
    # and the mean error of the five models
    f = open('dec_tree_data.pckl', 'wb')
    pickle.dump([best_dec_tree_model,
                 best_dec_tree_model_error,
                 error_dec_tree.mean()],
                f)
    f.close()

    return error_dec_tree;

def log_reg():
    max_c = 5
    error_per_c = np.empty((max_c, K_in))  # error per c for each k_in
    avg_error_c = np.empty(max_c)  # average error per c
    best_c = np.zeros(K_out)    # best error per k_out
    error_log_reg = np.empty(K_out)  # error per model

    best_log_reg_model_error = 1;
    best_log_reg_model_weight = np.zeros(numAttributes);

    # Outer K-fold crossvalidation
    CV_out = model_selection.KFold(n_splits=K_out, shuffle=True)

    k_out = 0
    print('===========================')
    print('MULTINOMIAL LOGISTIC REGRESSION')
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

            for c_index in range(0, max_c):
                # Fit classifier and classify the test points
                log_reg = lm.LogisticRegression(C=c_index+1, solver='lbfgs', multi_class='multinomial')
                log_reg.fit(X_train_in, Y_train_in.ravel())
                y_est_test_in = log_reg.predict(X_test_in);

                # find estimated test error for each X_test_in value
                error_y_est_test_in = sum(np.abs(y_est_test_in - Y_test_in)) / float(len(y_est_test_in))
                # find average estimated test error per depth for each k_out
                error_per_c[c_index, k_out] = error_y_est_test_in.mean()
            k_in += 1

        # we find the mean error per C, and we take the C with the minimum error
        minError = 1
        for c_index in range(0, max_c):
            avg_error_c[c_index] = error_per_c[c_index, :].mean()
            if avg_error_c[c_index] < minError:
                minError = avg_error_c[c_index]
                best_c[k_out] = int(c_index + 1);

        # we create a model using the best C and we find its error
        log_reg = lm.LogisticRegression(C=best_c[k_out], solver='lbfgs', multi_class='multinomial')
        log_reg.fit(X_train_out, Y_train_out.ravel())
        y_log_reg = log_reg.predict(X_test_out)

        # we calculate the error of the model
        error_log_reg[k_out] = (sum(np.abs(y_log_reg - Y_test_out)) / float(len(y_log_reg))).mean();

        # save the model with the lowest error
        if error_log_reg[k_out] <= best_log_reg_model_error:
            best_log_reg_model_error = error_log_reg[k_out]
            best_log_reg_model = log_reg
            best_log_reg_model_weight = log_reg.coef_

        # Display results
        print('    =============================')
        print('    Best C: {0}'.format(best_c[k_out]))
        print('    Error: {0}'.format(error_log_reg[k_out]))
        print('\n')
        k_out += 1

    # knowing the error for the 5 folds, we find the mean of the errors and this is the final error of the method
    print('==============================================')
    print('Best Model C = {0}'.format(best_log_reg_model.C))
    print('Best Model Error: {0}'.format(best_log_reg_model_error))
    print('Best Model Weights: {0}'.format(best_log_reg_model_weight))
    print('==============================================')
    print('Multinomial Regression Error: {0}'.format(error_log_reg.mean()))
    print('==============================================')
    print('\n')

    # save the best model, the best model weights, the best model error,
    # and the mean error of the five models
    f = open('log_reg_data.pckl', 'wb')
    pickle.dump([best_log_reg_model,
                 best_log_reg_model_weight,
                 best_log_reg_model_error,
                 error_log_reg.mean()],
                f)
    f.close()

    return error_log_reg

def knn():
    max_l = 50
    error_per_l = np.empty((max_l, K_in))  # error per lambda for each k_in
    avg_error_l = np.empty(max_l)  # average error per lambda
    best_l = np.empty(K_out)    # best error per k_out
    error_knn = np.empty(K_out)  # error per model

    best_knn_model_error = 1;

    # Outer K-fold crossvalidation
    CV_out = model_selection.KFold(n_splits=K_out, shuffle=True)

    k_out = 0
    print('===========================')
    print('   K-NEAREST NEIGHBOURS')
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

            for l_index in range(0, max_l):
                # Fit classifier and classify the test points
                knn = KNeighborsClassifier(n_neighbors=l_index+1)
                knn.fit(X_train_in, Y_train_in.ravel());
                y_est_test_in = knn.predict(X_test_in);

                # find estimated test error for each X_test_in value
                error_y_est_test_in = sum(np.abs(y_est_test_in - Y_test_in)) / float(len(y_est_test_in))
                # find average estimated test error per depth for each k_out
                error_per_l[l_index, k_out] = error_y_est_test_in.mean()
            k_in += 1

        # we find the mean error per C, and we take the C with the minimum error
        minError = 1
        for l_index in range(0, max_l):
            avg_error_l[l_index] = error_per_l[l_index, :].mean()
            if avg_error_l[l_index] < minError:
                minError = avg_error_l[l_index]
                best_l[k_out] = l_index + 1;

        # we create a model using the best lambda and find its error
        knn = KNeighborsClassifier(n_neighbors=int(best_l[k_out]))
        knn = knn.fit(X_train_out, Y_train_out.ravel())
        y_knn = knn.predict(X_test_out)

        # we calculate the final error of each model
        error_knn[k_out] = (sum(np.abs(y_knn - Y_test_out)) / float(len(y_knn))).mean();

        # save the model with the lowest error
        if error_knn[k_out] <= best_knn_model_error:
            best_knn_model_error = error_knn[k_out]
            best_knn_model = knn

        # Display results
        print('    =============================')
        print('    Best Num of Neighbours: {0}'.format(best_l[k_out]))
        print('    Error: {0}'.format(error_knn[k_out]))
        print('\n')
        k_out += 1
    #
    # knowing the error for the 5 folds, we find the mean of the errors and this is the final error of the method
    print('==============================================')
    print('Best Model n_neighbours = {0}'.format(best_knn_model.n_neighbors))
    print('Best Model Error: {0}'.format(best_knn_model_error))
    print('==============================================')
    print('K-Nearest Neighbour Error: {0}'.format(error_knn.mean()))
    print('==============================================')
    print('\n')

    # save the best model, the best model error,
    # and the mean error of the five models
    f = open('knn_data.pckl', 'wb')
    pickle.dump([best_knn_model,
                 best_knn_model_error,
                 error_knn.mean()],
                f)
    f.close()

    return error_knn

# [1] ##########################################################################
# In this problem we are trying to predict one class between
# 1,2,3,4 and 5 having known 10 attributes

# [2]
error_dec_tree = np.empty(K_out)
error_logreg = np.empty(K_out)
error_knn = np.empty(K_out)

error_dec_tree = dec_tree() # run decision tree
error_logreg = log_reg()      # run multinomial logistic regression
error_knn = knn()           # run k-nearest neighbour

# [3]
# Export tree graph for visualization purposes:
# out = tree.export_graphviz(dtc, out_file='tree_deviance.gvz', feature_names=attributeNames)
# src = graphviz.Source.from_file('tree_deviance.gvz')
# src.render('../tree_deviance', view=True)

# Get weights of the last logistic regression model













# Boxplot to compare classifier error distributions
# figure()
# boxplot(np.concatenate((error_dec_tree.T, error_logreg.T, error_knn.T), axis=1))
# xlabel('Decision Tree vs Logistic Regression vs K-Nearest Neighbour')
# ylabel('Cross-validation error [%]')
# show()