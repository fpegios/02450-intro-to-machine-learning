###############################################################
# CLASSIFICATION PROBLEM
###############################################################
import numpy as np
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier
import pickle
from _load_data import *

# Initialize variables
K_out = 5
K_in = 10
# Outer K-fold crossvalidation
CV_out = model_selection.KFold(n_splits=K_out, shuffle=True)
error_weight = np.empty(K_out);

# Decision Tree Variables
max_depth = 20
criterion = 'gini'
tree_complexity = np.arange(2, max_depth + 1, 1)  # Tree complexity parameter - constraint on maximum depth
error_per_depth = np.empty((max_depth - 1, K_in))  # error per depth for each k_in
avg_error_depth = np.empty(max_depth - 1)  # average error per depth
best_depth = np.empty(K_out)    # best error per k_out
test_error_dec_tree = np.empty(K_out)  # test error per model
best_dec_tree_model_error = 1;

# Logistic Regression Variables
max_c = 5
error_per_c = np.empty((max_c, K_in))  # error per c for each k_in
avg_error_c = np.empty(max_c)  # average error per c
best_c = np.zeros(K_out)    # best error per k_out
test_error_log_reg = np.empty(K_out)  # error per model
best_log_reg_model_error = 1;
best_log_reg_model_weight = np.zeros(numAttributes + 1);

# KNN Variables
max_l = 250
error_per_l = np.empty((max_l, K_in))  # error per lambda for each k_in
avg_error_l = np.empty(max_l)  # average error per lambda
best_l = np.empty(K_out)    # best error per k_out
test_error_knn = np.empty(K_out)  # error per model
best_knn_model_error = 1;

k_out = 0
for train_out_index, test_out_index in CV_out.split(X):
    print('\nComputing CV_out fold: {0}/{1}..'.format(k_out + 1, K_out))

    # extract training and test set for current CV_out fold
    X_train_out, Y_train_out = X[train_out_index,:], Y[train_out_index]
    X_test_out, Y_test_out = X[test_out_index,:], Y[test_out_index]
    error_weight[k_out] = len(X_test_out)/(len(X_test_out)*K_out);

    # Inner K-fold crossvalidation
    CV_in = model_selection.KFold(n_splits=K_in, shuffle=True)

    k_in = 0
    for train_in_index, test_in_index in CV_in.split(X_train_out):
        print('    Computing CV_in fold: {0}/{1}..'.format(k_in + 1, K_in))

        # extract training and test set for current CV_in fold
        X_train_in, Y_train_in = X_train_out[train_in_index, :], Y_train_out[train_in_index]
        X_test_in, Y_test_in = X_train_out[test_in_index, :], Y_train_out[test_in_index]

        # DECISION TREE
        for depth_index, depth in enumerate(tree_complexity):
            # Fit decision tree classifier with different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion=criterion, max_depth=depth)
            dtc = dtc.fit(X_train_in, Y_train_in.ravel())
            y_est_test_in = dtc.predict(X_test_in)

            # find estimated test error for each X_test_in value
            error_y_est_test_in = sum(np.abs(y_est_test_in - Y_test_in)) / float(len(y_est_test_in))
            # find average estimated test error per depth for each k_out
            error_per_depth[depth_index, k_in] = error_y_est_test_in.mean()

        # LOGISTIC REGRESSION
        for c_index in range(0, max_c):
            # Fit classifier and classify the test points
            log_reg = lm.LogisticRegression(C=c_index + 1, solver='lbfgs', multi_class='multinomial')
            log_reg.fit(X_train_in, Y_train_in.ravel())
            y_est_test_in = log_reg.predict(X_test_in);

            # find estimated test error for each X_test_in value
            error_y_est_test_in = sum(np.abs(y_est_test_in - Y_test_in)) / float(len(y_est_test_in))
            # find average estimated test error per depth for each k_out
            error_per_c[c_index, k_in] = error_y_est_test_in.mean()

        # KNN
        for l_index in range(0, max_l):
            # Fit classifier and classify the test points
            knn = KNeighborsClassifier(n_neighbors=l_index + 1)
            knn.fit(X_train_in, Y_train_in.ravel());
            y_est_test_in = knn.predict(X_test_in);

            # find estimated test error for each X_test_in value
            error_y_est_test_in = sum(np.abs(y_est_test_in - Y_test_in)) / float(len(y_est_test_in))
            # find average estimated test error per depth for each k_out
            error_per_l[l_index, k_in] = error_y_est_test_in.mean()

        k_in += 1

    # DECISION TREE
    # =============
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
    test_error_dec_tree[k_out] = (sum(np.abs(y_dec_tree - Y_test_out)) / float(len(y_dec_tree))).mean();

    # save the model with the lowest error
    if test_error_dec_tree[k_out] <= best_dec_tree_model_error:
        best_dec_tree_model_error = test_error_dec_tree[k_out]
        best_dec_tree_model = dtc
        best_dec_tree_model_X_train = X_train_out
        best_dec_tree_model_Y_train = Y_train_out
        best_dec_tree_model_X_test = X_test_out
        best_dec_tree_model_Y_test = y_dec_tree

    print('\n=============================')
    print('DECISION TREE')
    print('Best Depth: {0}'.format(best_depth[k_out]))
    print('Error: {0}'.format(test_error_dec_tree[k_out]))
    print('=============================')

    # LOGISTIC REGRESSION
    # ===================
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
    test_error_log_reg[k_out] = (sum(np.abs(y_log_reg - Y_test_out)) / float(len(y_log_reg))).mean();

    # save the model with the lowest error
    if test_error_log_reg[k_out] <= best_log_reg_model_error:
        best_log_reg_model_error = test_error_log_reg[k_out]
        best_log_reg_model = log_reg
        best_log_reg_model_weight = log_reg.coef_
        best_log_reg_model_X_train = X_train_out
        best_log_reg_model_Y_train = Y_train_out
        best_log_reg_model_X_test = X_test_out
        best_log_reg_model_Y_test = y_log_reg

    print('=============================')
    print('LOGISTIC REGRESSION')
    print('Best C: {0}'.format(best_c[k_out]))
    print('Error: {0}'.format(test_error_log_reg[k_out]))
    print('=============================')

    # KNN
    # =============
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
    test_error_knn[k_out] = (sum(np.abs(y_knn - Y_test_out)) / float(len(y_knn))).mean();

    # save the model with the lowest error
    if test_error_knn[k_out] <= best_knn_model_error:
        best_knn_model_error = test_error_knn[k_out]
        best_knn_model = knn
        best_knn_model_X_train = X_train_out
        best_knn_model_Y_train = Y_train_out
        best_knn_model_X_test = X_test_out
        best_knn_model_Y_test = y_knn

    print('=============================')
    print('KNN')
    print('Best Num of Neighbours: {0}'.format(best_l[k_out]))
    print('Error: {0}'.format(test_error_knn[k_out]))
    print('=============================\n')

    k_out += 1

# Calculate the generalization error
gen_error_dec_tree = sum(error_weight * test_error_dec_tree)

print('\n')
print('==============================================')
print('    DECISION TREE')
print('==============================================')
print('Best Model Depth: {0}'.format(best_dec_tree_model.max_depth))
print('Best Model Error: {0}'.format(best_dec_tree_model_error))
print('==============================================')
print('Generalization Error: {0}'.format(gen_error_dec_tree))
print('==============================================')
print('\n')

# save the best tree, the best model error,
# and the errors of the five models
f = open('dec_tree_data.pckl', 'wb')
pickle.dump([best_dec_tree_model,
             best_dec_tree_model_error,
             best_dec_tree_model_X_train,
             best_dec_tree_model_Y_train,
             best_dec_tree_model_X_test,
             best_dec_tree_model_Y_test,
             test_error_dec_tree,
             gen_error_dec_tree],
            f)
f.close()

# Calculate the generalization error
gen_error_log_reg = sum(error_weight * test_error_log_reg)

print('==============================================')
print('    LOGISTIC REGRESSION')
print('==============================================')
print('Best Model C: {0}'.format(best_log_reg_model.C))
print('Best Model Error: {0}'.format(best_log_reg_model_error))
print('==============================================')
print('Generalization Error: {0}'.format(gen_error_log_reg))
print('==============================================')
print('\n')

# save the best model, the best model weights, the best model error,
# and errors of the five models
f = open('log_reg_data.pckl', 'wb')
pickle.dump([best_log_reg_model,
             best_log_reg_model_error,
             best_log_reg_model_weight,
             best_log_reg_model_X_train,
             best_log_reg_model_Y_train,
             best_log_reg_model_X_test,
             best_log_reg_model_Y_test,
             test_error_log_reg,
             gen_error_log_reg],
            f)
f.close()

# Calculate the generalization error
gen_error_knn = sum(error_weight * test_error_knn)

print('==============================================')
print('    KNN')
print('==============================================')
print('Best Model n_neighbours: {0}'.format(best_knn_model.n_neighbors))
print('Best Model Error: {0}'.format(best_knn_model_error))
print('==============================================')
print('Generalization Error: {0}'.format(gen_error_knn))
print('==============================================')
print('\n')

# save the best model, the best model error,
# and the errors of the five models
f = open('knn_data.pckl', 'wb')
pickle.dump([best_knn_model,
             best_knn_model_error,
             best_knn_model_X_train,
             best_knn_model_Y_train,
             best_knn_model_X_test,
             best_knn_model_Y_test,
             test_error_knn,
             gen_error_knn],
            f)
f.close()

# Boxplot to compare classifier error distributions
# figure()
# boxplot(np.concatenate((error_dec_tree.T, error_logreg.T, error_knn.T), axis=1))
# xlabel('Decision Tree vs Logistic Regression vs K-Nearest Neighbour')
# ylabel('Cross-validation error [%]')
# show()