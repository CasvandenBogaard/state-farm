import pandas as pd
from sklearn import cross_validation
import os.path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# reading the training data
class0 = pd.read_csv("activations/activations_train_c0.csv")
class1 = pd.read_csv("activations/activations_train_c1.csv")
class2 = pd.read_csv("activations/activations_train_c2.csv")
class3 = pd.read_csv("activations/activations_train_c3.csv")
class4 = pd.read_csv("activations/activations_train_c4.csv")
class5 = pd.read_csv("activations/activations_train_c5.csv")
class6 = pd.read_csv("activations/activations_train_c6.csv")
class7 = pd.read_csv("activations/activations_train_c7.csv")
class8 = pd.read_csv("activations/activations_train_c8.csv")
class9 = pd.read_csv("activations/activations_train_c9.csv")

# append the class label to the dataframes

class0['label'] = 0
class1['label'] = 1
class2['label'] = 2
class3['label'] = 3
class4['label'] = 4
class5['label'] = 5
class6['label'] = 6
class7['label'] = 7
class8['label'] = 8
class9['label'] = 9

# concatenate the dataframes

data = pd.concat((class0,class1,class2,class3,class4,class5,class6,class7,class8,class9))
N = data.shape[0]

#prepare training and testig for random forrest classifier
# todo split on drivers!

# cross_validation
K = 10
avg_loss = 0
kf = cross_validation.KFold(N, n_folds=K, shuffle=True)
for train, test in kf:
    train_x = data.iloc[train, 0:8192]
    train_y = data['label'].iloc[train]
    test_x = data.iloc[test, 0:8192]
    test_y = data['label'].iloc[test]

    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    clf.fit(train_x, train_y)
    predicted_y = clf.predict_proba(test_x)
    loss = log_loss(test_y, predicted_y)
    avg_loss += loss
    print "Log-loss: " + str(loss)

final_loss = avg_loss/K
print "Average logg-loss: " + str(final_loss)