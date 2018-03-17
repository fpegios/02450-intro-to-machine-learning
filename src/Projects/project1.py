import numpy as np

import os
import scipy.linalg as linalg
from categoric2numeric import categoric2numeric
from matplotlib.pyplot import (figure, subplot, plot, xlabel, hist, ylabel, title,
                               xticks, yticks, show, legend, imshow, xlim, subplots, scatter, text, cm, savefig)
from mpl_toolkits.mplot3d import Axes3D

# %% Marketing

# Load data
filename = "marketingdata.txt"
marketingdata = np.genfromtxt(filename, delimiter=" ")
N = len(marketingdata)
from collections import Counter

# Remove nans
count = 0
for i in range(N):
    if True in np.isnan(marketingdata[i, :]):
        count += 1

sortedData = np.zeros((N - count, 14))
Count = 0
for i in range(N):
    if True not in np.isnan(marketingdata[i, :]):
        sortedData[Count, :] = marketingdata[i, :]
        Count += 1

# The new data
X = np.mat(sortedData[:, 1:])
y = np.mat(sortedData[:, 0]).T
# X = np.hstack((np.mat(sortedData[:,0:]) , np.mat(sortedData[:,2:])))
# y = np.mat(sortedData[:,1]).T
N = len(y)

# OOK-Coding
nul = categoric2numeric(X[:, 0])[0] / np.sqrt(X[:, 0].max())
et = categoric2numeric(X[:, 1])[0] / np.sqrt(X[:, 1].max())
# to   = X[:,2]
to = categoric2numeric(X[:, 2])[0] / np.sqrt(X[:, 2].max())
tre = categoric2numeric(X[:, 3])[0] / np.sqrt(X[:, 3].max())
fire = categoric2numeric(X[:, 4])[0] / np.sqrt(X[:, 4].max())
fem = categoric2numeric(X[:, 5])[0] / np.sqrt(X[:, 5].max())
seks = categoric2numeric(X[:, 6])[0] / np.sqrt(X[:, 6].max())
# syv  = X[:,7]
syv = categoric2numeric(X[:, 7])[0] / np.sqrt(X[:, 7].max())
# otte = X[:,8]
otte = categoric2numeric(X[:, 8])[0] / np.sqrt(X[:, 8].max())
ni = categoric2numeric(X[:, 9])[0] / np.sqrt(X[:, 9].max())
ti = categoric2numeric(X[:, 10])[0] / np.sqrt(X[:, 10].max())
elleve = categoric2numeric(X[:, 11])[0] / np.sqrt(X[:, 11].max())
tolv = categoric2numeric(X[:, 12])[0] / np.sqrt(X[:, 12].max())
X1 = np.asarray(np.hstack((nul, et, to, tre, fire, fem, seks, syv, otte, ni, ti, elleve, tolv)))
# X1 = np.asarray(np.hstack((to,tre,fem,syv,otte)))
# X1 = np.asarray(np.hstack((et,seks,ti,tolv)))

# Standardizing
# Xst = (X1 - np.ones((N,1))*X1.mean(0)) * 1 / (np.ones((N,1))*X1.std(0))
Xst = X1  # - np.ones((N,1))*X1.mean(0)
N, M = Xst.shape

# PCA by computing SVD of Y
U, S, V = linalg.svd(Xst, full_matrices=False)
V = V.T

# De vigtigste attributer for de første PCA's
print(np.where(abs(V[:, 0]).max() == abs(V[:, 0]))[0][0])
print(np.where(abs(V[:, 1]).max() == abs(V[:, 1]))[0][0])
print(np.where(abs(V[:, 2]).max() == abs(V[:, 2]))[0][0])
print(np.where(abs(V[:, 3]).max() == abs(V[:, 3]))[0][0])
print(np.where(abs(V[:, 4]).max() == abs(V[:, 4]))[0][0])

# attribute 2 (married) is most important for PC1, attribute 22 (Factory Worker/Laborer/Driver)
# is most important for PC2 and attribute 40 (four persons in household) is most important for PC3
#
#

# eller hvis OOK encoding er lavet på alle: 6 for PC1, 57 for PC2 og 27 for PC3
# 72 for PC4 I.e. the attributes, 1) being single, 2) rent
# 3) retired 4) speaking spanish means the most? The task is to predict the annual income.
# I guess that makes sense?

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

# 47 PC's forklarer 90% af variansen rho[0:47].sum()

# Plot variance explained
figure()
plot(rho, 'o-')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained value');

# Project data onto principal component space
Z = Xst @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data

n = [1, 9]
colors = ['blue', 'green', 'red']
classValues = n
# classNames = [str(num) for num in n]
# classNames = ['<10K','10-14K','15-19K','20-24K','25-29K','30-39K','40-49K','50-75K','75K+']
classNames = ['<10K', '75K+']
classDict = dict(zip(classNames, classValues))
C = len(classNames)

figure()
for c in n:
    # select indices belonging to class c:
    class_mask = (np.asarray(y.T)[0] == c)
    # Plotter de projekterede billeder for tallene 0 og 1
    plot(Z[class_mask, 0], Z[class_mask, 1], 'o')  # 0 er PC1 og 1 er PC2
legend(classNames)
xlabel('PC1')
ylabel('PC2')

# %%

titles = ['Sex', 'Marital Status', 'Age', 'Education', 'Occupation', 'Lived in area', 'Dual incomes',
          'Persons in house', 'Persons in house under 18', 'Householder status', 'Type of home',
          'Ethnic classification', 'Language']
for i in range(np.shape(X)[1]):

    data = np.asarray(X[:, i].T)[0].astype(int)
    if i == 8:
        continue
    counts = np.bincount(data)
    fig, ax = subplots()
    ax.bar(range(data.max() + 1), counts, width=0.8, align='center')
    ax.set(xticks=range(data.max() + 1), xlim=[0, data.max() + 3])
    # labels = [0,'Mænd', 'Kvinder']
    # ax.set_xticklabels(labels)
    # legend(['hej', 'hejsa'])
    title(titles[i])
    ylabel("Antal")
    # show()

data = np.asarray((X[:, 8] + 1).T)[0].astype(int)
counts = np.bincount(data)
fig, ax = subplots()
ax.bar(range(data.max() + 1), counts, width=0.8, align='center')
ax.set(xticks=range(data.max() + 1), xlim=[0, data.max() + 1])
# labels = [0,'Mænd', 'Kvinder']
# ax.set_xticklabels(labels)
title(titles[8])
ylabel("Antal")
show()

data = np.asarray(y.T)[0].astype(int)
counts = np.bincount(data)
fig, ax = subplots()
ax.bar(range(data.max() + 1), counts, width=0.8, align='center')
ax.set(xticks=range(data.max() + 1), xlim=[0, data.max() + 1])
# labels = [0,'Mænd', 'Kvinder']
# ax.set_xticklabels(labels)
title("Annual income")
ylabel("Antal")
show()

# %%

Ylabel = "Dual incomes"
# x = ['14-17','18-24','25-34','35-44','45-54','55-65','65+']
# x = ['Male','Female']
# x = ['<Grade 8','Grade\n9-11','High\nSchool\nGrad','1-3 years\ncollege','College\nGrad','Grad\nStudy']
# x = ['Mana-\ngerial','Sales\nWorker','Factory\nWorker','Service\nWorker','Home\nmaker','Student','Military','Retired','Unem-\nployed']
# x = ['Married','Roomies','Divorced\nSeparated','Widowed','Single']
# x = ['<1','1-3','4-6','7-10','>10']
x = ['Not\nMarri-\ned', 'Yes', 'No']
# x = ['1','2','3','4','5','6','7','8','9+']
# x = ['0','1','2','3','4','5','6','7','8','9+',]
# x = ['Own','Rent','With\nfam']
# x = ['House','Condo','Apart','Mobile','Other']
# x = ['Am.\nInd.','Asian','Black','East\nInd','His-\npanic','Pac.\nIsl.','White','Other']
# x = ['<10','10-14','15-19','20-24','25-29','30-39','40-49','50-75','75+']
# x = ['Engl.','Span.','Other']
ting = np.asarray(X[:, 6].T)[0]

values = np.array([])
for i in range(0, int(ting.max())):
    value = Counter(ting)[i + 1]
    values = np.append(values, value)

fig, ax = subplots()
# width = 0.75 # the width of the bars
xlabel('Number')
ylabel(Ylabel)
title("Bar chart of " + Ylabel)
ind = np.arange(len(values))  # the x locations for the groups
xlim(0, values.max() + 1 / 6 * values.max())
ax.barh(ind, values, color="blue")
ax.set_yticks(ind)
ax.set_yticklabels(x, minor=False)
for i, v in enumerate(values):
    ax.text(v + 50, i - 0.1, str(round(v / N * 100, 1)) + "%", color='blue', fontweight='bold')

savefig(os.path.join(Ylabel + ".png"))

# %%

attributeNames = ['Sex', 'Marital status', 'Age', 'Education', 'Occupation', 'Lived in area', 'Dual income',
                  'Persons over 18', 'Under 18', 'Householder status', 'Type of home', 'Ethnic', 'Language']
Names = [attributeNames[1], attributeNames[6], attributeNames[10], attributeNames[12]]
Matrix = np.hstack((X[:, 1], X[:, 6], X[:, 10], X[:, 12]))

for j in range(0, 5):
    for k in range(0, 5):
        figure()
        A = np.asarray(X[:, j].T.copy())[0]
        B = np.asarray(X[:, k].T.copy())[0]
        for i in range(len(A)):
            A[i] = A[i] + np.random.uniform(-.1, .1, 1)[0]
            B[i] = B[i] + np.random.uniform(-.1, .1, 1)[0]
        scatter(A, B)
        xlabel(attributeNames[j])
        ylabel(attributeNames[k])

# There are no correlation but we can see unlikely possibilities
# %%

fig, ax = subplots()
A = np.asarray(X[:, 2].T.copy())[0]
B = np.asarray(X[:, 1].T.copy())[0]
for i in range(len(A)):
    A[i] = A[i] + np.random.uniform(-.1, .1, 1)[0]
    B[i] = B[i] + np.random.uniform(-.1, .1, 1)[0]
scatter(A, B)
ys = ['Married', 'Roomies', 'Divorced\nSeparated', 'Widowed', 'Single']
group_labels = ['14-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
xlabel('Age')
ylabel('Marital Status')
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_xticklabels(group_labels, minor=False)
ax.set_yticklabels(ys, minor=False)
# xlabel(attributeNames[2])
# ylabel(attributeNames[1])
