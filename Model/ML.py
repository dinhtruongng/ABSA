import pandas as pd 
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
from pyvi import ViTokenizer, ViPosTagger
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC


with open ('/Users/khangdoan/Documents/GitHub/ABSA/Model/y', 'rb') as f:
    y = pickle.load(f)

with open ('/Users/khangdoan/Documents/GitHub/ABSA/Model/BoW_for_ML', 'rb') as f:
    BoW = pickle.load(f)

x_train,x_test,y_train,y_test = train_test_split(BoW,np.asarray(y))

# param = {
#     'estimator__C': [0.1, 1, 10, 100],
#     'estimator__gamma': [1, 0.1, 0.01, 0.001],
#     'estimator__kernel': ['rbf', 'linear']
# }
# This is for the hyperparameter tuning
# model = MultiOutputClassifier(SVC())

# grid = GridSearchCV(model, param, cv = 2) 

# clf = grid.fit(x_train, y_train)
# res = pd.DataFrame(grid.cv_results_)
# best_para = clf.best_params_
new_clf = MultiOutputClassifier(SVC(C=10, gamma=0.01, kernel='rbf')).fit(x_train, y_train)
predictions = new_clf.predict(x_test)

model_path = "/Users/khangdoan/Documents/GitHub/ABSA/Model/ML_Model"
with open (model_path, 'wb') as f:
    pickle.dump(new_clf, f)
