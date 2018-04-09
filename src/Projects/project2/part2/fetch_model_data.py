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

# print(dec_tree[0])
# print("\n")
# print(log_reg[0])
# print("\n")
# print(knn[0])

dec_tree_gen_error = dec_tree[len(dec_tree) - 1]
log_reg_gen_error = log_reg[len(log_reg) - 1]
knn_gen_error = knn[len(knn) - 1]

dec_tree_test_error = dec_tree[len(dec_tree) - 2]
log_reg_test_error = log_reg[len(log_reg) - 2]
knn_test_error = knn[len(knn) - 2]

def dec_tree_graph():
    out = tree.export_graphviz(dec_tree[0], out_file='tree_deviance.gvz', feature_names=attributeNames)
    src = graphviz.Source.from_file('tree_deviance.gvz')
    src.render('../tree_deviance', view=True)

def knn_plot():
    knn_x_train = knn[2]
    knn_y_train = knn[3]
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
    attr_1 = 10;
    attr_2 = 6;

    # KNN
    # Plot the training data points (color-coded) and test data points.
    figure(1)
    for i in range(len(knn_y_train_styles)):
        plot(knn_x_train[i, attr_1], knn_x_train[i, attr_2], knn_y_train_styles[i])

    for i in range(len(knn_y_test_styles)):
        plot(knn_x_test[i, attr_1], knn_x_test[i, attr_2], knn_y_test_styles[i], markersize=10)
        plot(knn_x_test[i, attr_1], knn_x_test[i, attr_2], 'kx', markersize=8)
    show()

# dec_tree_graph()
# knn_plot()

print("\n")
print("==============================================")
print("     GENERALIZATION ERROR")
print("==============================================")
print("Decision Tree: ", format(dec_tree_gen_error))
print("Logistic Regression: ", format(log_reg_gen_error))
print("KNN: ", format(knn_gen_error))
print("==============================================")
print("\n")

# [4.1] Credibility Interval
# Test if classifiers are significantly different by computing credibility interval.
temp = np.zeros([len(log_reg_test_error), 1])
temp[:, 0] = log_reg_test_error[:];
log_reg_test_error = temp;

temp = np.zeros([len(knn_test_error), 1])
temp[:, 0] = knn_test_error[:];
knn_test_error = temp;

z = (log_reg_test_error - knn_test_error)
zb = z.mean()
K = 5
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

print("==============================================")
print("     CREDIBILITY INTERVAL")
print("==============================================")
if zL <= 0 and zH >= 0:
    print('Classifiers are not significantly different')
else:
    print('Classifiers are significantly different.')
print("==============================================")

# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((log_reg_test_error, knn_test_error), axis=1))
xlabel('Multinomial Logistic Regression    vs    KNN')
ylabel('Cross-validation error')

show()

# [4.2] Compare in addition if the performance of your models
# are better than simply predicting all outputs to be the largest
# class in the training data
dec_tree_best_model_error = dec_tree[1]
y_train = dec_tree[3];
numLargestClass = 0
for i in range(len(y_train)):
    if y_train[i] == 1: # class 1 (text) is the largest one
        numLargestClass+=1;

largestClassProbability = numLargestClass/len(y_train)
dec_tree_accuracy = 1 - dec_tree_best_model_error;

log_reg_best_model_error = log_reg[1]
y_train = log_reg[4];
numLargestClass = 0
for i in range(len(y_train)):
    if y_train[i] == 1:  # class 1 (text) is the largest one
        numLargestClass+=1;

largestClassProbability = numLargestClass/len(y_train)
log_reg_accuracy = 1 - log_reg_best_model_error;


knn_best_model_error = knn[1]
y_train = knn[3];
numLargestClass = 0
for i in range(len(y_train)):
    if y_train[i] == 1: # class 1 (text) is the largest one
        numLargestClass+=1;

largestClassProbability = numLargestClass/len(y_train)
knn_accuracy = 1 - knn_best_model_error;

print("==============================================")
print("Best Dec Tree Accuracy: ", format(dec_tree_accuracy))
print("Logistic Regression Model Accuracy: ", format(log_reg_accuracy))
print("Best KNN Model Accuracy: ", format(knn_accuracy))
print("Largest Class Probability: ", format(largestClassProbability))
print("==============================================")