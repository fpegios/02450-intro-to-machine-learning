import pickle
from matplotlib.pyplot import (figure, hold, plot, title, xlabel, ylabel,
                               colorbar, imshow, xticks, yticks, boxplot, show)
from sklearn import tree
import graphviz
from scipy import stats

from _load_data import *

# fetched saved model data
dec_tree_f = open('dec_tree_data.pckl', 'rb')
dec_tree = pickle.load(dec_tree_f)
dec_tree_f.close()
log_reg_f = open('log_reg_data.pckl', 'rb')
log_reg = pickle.load(log_reg_f)
log_reg_f.close()
knn_f = open('knn_data.pckl', 'rb')
knn = pickle.load(knn_f)
knn_f.close()

dec_tree_error = dec_tree[len(dec_tree) - 1]
knn_error = knn[len(knn) - 1]

def knn_plot():
    knn_x_train = knn[2];
    knn_y_train = knn[3];
    knn_x_test = knn[4]
    knn_y_test = knn[5]

    styles_dot = ['.b', '.r', '.g', '.y', '.m']
    styles_circle = ['ob', 'or', 'og', 'oy', 'om']
    knn_y_train_styles = ["" for x in range(len(knn_y_train))]
    knn_y_test_styles = ["" for x in range(len(knn_y_test))]

    # save colors per output
    for i in range(len(knn_y_train_styles)):
        if knn_y_train[i] == 1:
            knn_y_train_styles[i] = styles_dot[0]
        elif knn_y_train[i] == 2:
            knn_y_train_styles[i] = styles_dot[1]
        elif knn_y_train[i] == 3:
            knn_y_train_styles[i] = styles_dot[2]
        elif knn_y_train[i] == 4:
            knn_y_train_styles[i] = styles_dot[3]
        elif knn_y_train[i] == 5:
            knn_y_train_styles[i] = styles_dot[4]

    for i in range(len(knn_y_test_styles)):
        if knn_y_test[i] == 1:
            knn_y_test_styles[i] = styles_circle[0]
        elif knn_y_test[i] == 2:
            knn_y_test_styles[i] = styles_circle[1]
        elif knn_y_test[i] == 3:
            knn_y_test_styles[i] = styles_circle[2]
        elif knn_y_test[i] == 4:
            knn_y_test_styles[i] = styles_circle[3]
        elif knn_y_test[i] == 5:
            knn_y_test_styles[i] = styles_circle[4]

    # attributes plot
    attr_1 = 9;
    attr_2 = 5;

    # KNN
    # Plot the training data points (color-coded) and test data points.
    figure(1)
    for i in range(len(knn_y_train_styles)):
        plot(knn_x_train[i, attr_1], knn_x_train[i, attr_2], knn_y_train_styles[i])

    for i in range(len(knn_y_test_styles)):
        plot(knn_x_test[i, attr_1], knn_x_test[i, attr_2], knn_y_test_styles[i], markersize=10)
        plot(knn_x_test[i, attr_1], knn_x_test[i, attr_2], 'kx', markersize=8)
    show()

def dec_tree_graph():
    out = tree.export_graphviz(dec_tree[0], out_file='tree_deviance.gvz', feature_names=attributeNames)
    src = graphviz.Source.from_file('tree_deviance.gvz')
    src.render('../tree_deviance', view=True)

# eqs = log_reg[1];
# eq1 = eqs[0]
# eq2 = eqs[1]
# eq3 = eqs[2]
# eq4 = eqs[3]
# eq5 = eqs[4]
#
# out = np.zeros(5)
# print(eq1)
# pos = 5470
# out[0] = X[pos,0] * eq1[0] + X[pos,1] * eq1[1] + X[pos,2] * eq1[2] + X[pos,3] * eq1[3] + X[pos,4] * eq1[4] + X[pos,5] * eq1[5] + X[pos,6] * eq1[6] + X[pos,7] * eq1[7] + X[pos,8] * eq1[8] + X[pos,9] * eq1[9];
# out[1] = X[pos,0] * eq2[0] + X[pos,1] * eq2[1] + X[pos,2] * eq2[2] + X[pos,3] * eq2[3] + X[pos,4] * eq2[4] + X[pos,5] * eq2[5] + X[pos,6] * eq2[6] + X[pos,7] * eq2[7] + X[pos,8] * eq2[8] + X[pos,9] * eq2[9];
# out[2] = X[pos,0] * eq3[0] + X[pos,1] * eq3[1] + X[pos,2] * eq3[2] + X[pos,3] * eq3[3] + X[pos,4] * eq3[4] + X[pos,5] * eq3[5] + X[pos,6] * eq3[6] + X[pos,7] * eq3[7] + X[pos,8] * eq3[8] + X[pos,9] * eq3[9];
# out[3] = X[pos,0] * eq4[0] + X[pos,1] * eq4[1] + X[pos,2] * eq4[2] + X[pos,3] * eq4[3] + X[pos,4] * eq4[4] + X[pos,5] * eq4[5] + X[pos,6] * eq4[6] + X[pos,7] * eq4[7] + X[pos,8] * eq4[8] + X[pos,9] * eq4[9];
# out[4] = X[pos,0] * eq5[0] + X[pos,1] * eq5[1] + X[pos,2] * eq5[2] + X[pos,3] * eq5[3] + X[pos,4] * eq5[4] + X[pos,5] * eq5[5] + X[pos,6] * eq5[6] + X[pos,7] * eq5[7] + X[pos,8] * eq5[8] + X[pos,9] * eq5[9];

# eq1 = log_reg

# [3]
# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
z = (dec_tree_error - knn_error)
zb = z.mean()
K = 5
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0:
    print('Classifiers are not significantly different')
else:
    print('Classifiers are significantly different.')

# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((dec_tree_error.T, knn_error.T),axis=1))
xlabel('Decision Tree vs KNN')
ylabel('Cross-validation error')

show()