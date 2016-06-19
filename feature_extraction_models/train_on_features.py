import pandas as pd
from sklearn import cross_validation
import os.path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import minimize
import matplotlib.pyplot as plt






test_data = pd.read_csv("activations/activations_test.csv", chunksize=5000)

#Create the DataFrame
def get_driver_data():
    dr = dict()
    path = os.path.join('driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr

#drivers = get_driver_data()



data = pd.DataFrame()
for i in range(10):
    new_df = pd.read_csv('activations/activations_train_c{}.csv'.format(i))
    new_df_test = pd.read_csv('test_activations/test_train_c{}.csv'.format(i))

    new_df['label'] = i
    new_df_test['label'] =i


    #new_df['driver_id'] = [drivers[x] for x in new_df['img']]
    data = data.append(new_df, ignore_index=True)
    data = data.append(new_df_test, ignore_index = True)
print "data loaded"

# normalizing the data
#extra = pd.DataFrame()
#extra['mean'] = data.iloc[:,0:8192].mean(axis = 1)
#extra['var'] = data.iloc[:,0:8192].var(axis = 1)
#extra['zeros'] = (data.iloc[:,0:8192] == 0).sum(axis = 1)

#extra['mean'] = extra['mean'].apply(lambda x: (x - np.mean(x)) / (np.std(x)+0.001))
#extra['var'] = extra['var'].apply(lambda x: (x - np.mean(x)) / (np.std(x)+0.001))
#extra['zeros'] = extra['zeros'].apply(lambda x: (x - np.mean(x)) / (np.std(x)+0.001))


#print data.iloc[0,:]
#exit()
new_data=data.iloc[:,0:8192].apply(lambda x: (x - np.mean(x)) / (np.std(x)+0.001))
print "data normalized"



labels = data['label'].values
imgs = data['img'].values
#driver_id = data['driver_id'].values
#unique_drivers = list(set(driver_id))




#cross_validation
# K = 10
# avg_loss = 0
# kf = cross_validation.LabelKFold(unique_drivers, n_folds=K)
#
# #clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# #clf = KNeighborsClassifier(n_neighbors=5, n_jobs = -1)
# clf = LogisticRegression(penalty='l2', random_state=2016, solver='lbfgs', max_iter=5, multi_class='multinomial', warm_start=False, n_jobs=-1)
#
# clf_isotonic = CalibratedClassifierCV(clf, cv=4, method='sigmoid')
# for train_idx, test_idx in kf:
#     train_drivers = np.array(unique_drivers)[train_idx]
#     train_indices = [i for i, x in enumerate(list(driver_id)) if x in train_drivers]
#     test_drivers = np.array(unique_drivers)[test_idx]
#     test_indices = [i for i, x in enumerate(list(driver_id)) if x in test_drivers]
#
#
#     train_x = data.iloc[train_indices, 0:8192]
#     train_y = data['label'].iloc[train_indices]
#     test_x = data.iloc[test_indices,0:8192] # 4096
#     test_y = data['label'].iloc[test_indices]
#
#     clf_isotonic.fit(train_x, train_y)
#     predicted_y = clf_isotonic.predict_proba(test_x)
#
#
#
#     #clf.fit(train_x, train_y)
#     #predicted_y = clf.predict_proba(test_x)
#
#
#     loss = log_loss(test_y, predicted_y)
#     avg_loss += loss
#     print "Log-loss: " + str(loss)
#
# final_loss = avg_loss/K
# print "Average logg-loss: " + str(final_loss)





train_x= new_data.iloc[:, 0:8192]
#train_x = pd.concat([new_data,extra], axis = 1)
print train_x.shape
train_y = data['label']

print "starting training"
#clf = RandomForestClassifier(n_estimators=600, n_jobs=-1)
# newton-cg
#lbfgs
clf = LogisticRegression(penalty='l2', solver='newton-cg', max_iter=2, multi_class='multinomial', warm_start=True, n_jobs=-1)
#clf_isotonic = CalibratedClassifierCV(clf, cv=4, method='sigmoid')
clf.fit(train_x,train_y)
print "finished training!"

idx = 0
for chunks in test_data:
    #extra_test = pd.DataFrame()
    #extra_test['mean'] = chunks.iloc[:, 4096:8192].mean(axis=1)
    #extra_test['var'] = chunks.iloc[:, 4096:8192].var(axis=1)
    #extra_test['zeros'] = (chunks.iloc[:, 4096:8192] == 0).sum(axis=1)

    #extra_test['mean'] = extra_test['mean'].apply(lambda x: (x - np.mean(x)) / (np.std(x) + 0.001))
    #extra_test['var'] = extra_test['var'].apply(lambda x: (x - np.mean(x)) / (np.std(x) + 0.001))
    #extra_test['zeros'] = extra_test['zeros'].apply(lambda x: (x - np.mean(x)) / (np.std(x) + 0.001))

    test_x= chunks.iloc[:, 0:8192].apply(lambda x: (x - np.mean(x)) / (np.std(x)+0.001))
    #test_x = pd.concat([result_test,extra_test], axis = 1)
    test_id = chunks.iloc[:,8192]
    predictions = clf.predict_proba(test_x)

    # create submission
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    sub_file = os.path.join('subm', 'submission_' +str(idx)+'.csv')
    result1.to_csv(sub_file, index=False)
    print "saved " + str(idx)
    idx = idx +1






