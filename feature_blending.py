import pandas as pd
from sklearn import cross_validation

import os.path
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.linear_model import LogisticRegression


def get_driver_data():
    dr = dict()
    path = os.path.join('data', 'driver_imgs_list.csv')
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

def make_submission(y, imgs):
    labels = ['c0']
    df = pd.DataFrame(y)
    df['img'] = imgs
    
    df.to_csv('results/feature_submission.csv', index=False)
    return "Submission made"

def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data.iloc[i][:-3])
            target.append(train_target[i])
            index.append(i)
            
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index


#Create the training DataFrame
drivers = get_driver_data()

data = pd.DataFrame()
for i in range(10):
    new_df = pd.read_csv('activations/activations_train_c{}.csv'.format(i))
    new_df['label'] = i
    new_df['driver_id'] = [drivers[x] for x in new_df['img']]
    data = data.append(new_df, ignore_index=True)
    
labels = data['label'].values
imgs = data['img'].values
driver_id = data['driver_id'].values
unique_drivers = list(set(driver_id))


test_data = pd.read_csv('activations/activations_test.csv'.format(i))
test_imgs = test_data['img'].values


#KFold split on drivers
K = 10
kf = cross_validation.LabelKFold(unique_drivers, n_folds=K)
splits = []
for train_index, test_index in kf:
    train_drivers = np.array(unique_drivers)[train_index]
    test_drivers = np.array(unique_drivers)[test_index]
    splits.append((list(train_drivers),list(test_drivers)))


avg_loss = 0
for pair in splits:
    train_drivers = pair[0]
    test_drivers = pair[1]
    
    train_indices = [i for i,x in enumerate(list(driver_id)) if x in train_drivers]
    test_indices = [i for i,x in enumerate(list(driver_id)) if x in test_drivers]
    
    print train_indices
    
    train_x = data.iloc[train_indices, 0:8192]
    train_y = data['label'].iloc[train_indices]
    test_x = data.iloc[test_indices, 0:8192]
    test_y = data['label'].iloc[test_indices]
    
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(train_x, train_y)
    predicted_y = clf.predict_proba(test_x)
    loss = log_loss(test_y, predicted_y)
    avg_loss += loss
    print "Log-loss: " + str(loss)

final_loss = avg_loss/K
print "Average logg-loss: " + str(final_loss)



np.random.seed(0) # seed to shuffle the train set
verbose = True

#KFold split on drivers
K = 10
kf = cross_validation.LabelKFold(unique_drivers, n_folds=K)

splits = []
for train_index, test_index in kf:
    train_drivers = np.array(unique_drivers)[train_index]
    test_drivers = np.array(unique_drivers)[test_index]
    splits.append((list(train_drivers),list(test_drivers)))


clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
       ]

print "Creating train and test sets for blending."

dataset_blend_train = np.zeros((data.shape[0], len(clfs)))
dataset_blend_test = np.zeros((test_data.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((test_data.shape[0], len(kf)))
    for i, (train_drivers, test_drivers) in enumerate(splits):
        print "Fold", i
        train_indices = [k for k,x in enumerate(list(driver_id)) if x in train_drivers]
        test_indices = [k for k,x in enumerate(list(driver_id)) if x in test_drivers]
    
        X_train = data.iloc[train_indices, 0:8192]
        y_train = data['label'].iloc[train_indices]
        X_test = data.iloc[test_indices, 0:8192]
        y_test = data['label'].iloc[test_indices]

        clf.fit(X_train, y_train)
        y_submission = clf.predict(X_test)
        print y_submission.shape
        dataset_blend_train[test_indices, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict(X_submission)
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    

print
print "Blending."
clf = LogisticRegression()
clf.fit(dataset_blend_train, labels)
y_submission = clf.predict_proba(dataset_blend_test)[:,1]

print "Linear stretch of predictions to [0,1]"
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
