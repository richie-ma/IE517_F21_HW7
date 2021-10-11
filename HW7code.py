# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:56:21 2021

@author: ruchuan2
"""

#########  import the data #################

import pandas as pd


####################################################### Processing the data ############################################

data = pd.read_csv("C:/Users/ruchuan2/Box/IE 517 Machine Learning in FIN Lab/HW7/ccdefault.csv", header='infer')
del data["ID"]

#################### Random Forest

from sklearn.model_selection import train_test_split


# Split the dataset into a training and a testing set
### with loop

X, y = data.iloc[:,0:23].values, data.iloc[:,23]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

in_sample_cv_scores = []

start = time.time()

for i in range(1, 21):
    forest= RandomForestClassifier(n_estimators=i, criterion='gini', random_state=1, n_jobs=-1)
    cv_scores = cross_val_score(estimator = forest,
                                X=X_train,
                                y=y_train,
                                cv=10,
                                n_jobs=-1)
    mean_cv_scores = np.mean(cv_scores)
    in_sample_cv_scores.append(mean_cv_scores)

end=time.time()

run=end-start

####################################### importance #############################

forest= RandomForestClassifier(n_estimators=i, criterion='gini', random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
