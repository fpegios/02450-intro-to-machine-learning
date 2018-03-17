# Exercise 1
import numpy as np
import xlrd
from matplotlib.pyplot import (plot, figure, show, title)

# Load xls sheet with data
doc = xlrd.open_workbook('../Data/dataset.xls').sheet_by_index(0)

## 1.1
N = 10;

y = doc.col_values(0, 0, N);
print(y)
x = np.linspace(0, N-1, N)
print(x)

figure()
plot(x, y, 'o')

y = doc.col_values(1, 0, N);
figure()
plot(x, y, 'o')

show()

## 1.2
# Our data consists of 14 demographic attributes (mixture of categorical
# and continuous variables with a lot of missing data). With this data we
# The goal is to predict the Annual Income of Household from the other 13
# demographics  attributes.

## 1.3
# Given the 13 attributes we will use classification to find the class
# in which the relative income belongs to and regression in order to estimate the income.