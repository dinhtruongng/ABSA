import pandas as pd 
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
from pyvi import ViTokenizer, ViPosTagger
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC


with open ('/Users/khangdoan/Documents/GitHub/ABSA/Model/y', 'rb') as f:
    y = pickle.load(f)

with open ('/Users/khangdoan/Documents/GitHub/ABSA/Model/BoW_for_ML', 'rb') as f:
    BoW = pickle.load(f)

x_train,x_test,y_train,y_test = train_test_split(BoW,np.asarray(y))
# tuning hyperparameter
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


sen = predictions[:,0:10]
asp = predictions[:,10:20]

def calculate_macro_metrics(all_pred_categories, all_pred_sentiments, all_true_labels, num_aspect=10):
    """
    Calculate macro-averaged Precision, Recall, and F1-score for Aspect Detection and Sentiment Detection.

    Parameters:
    - all_pred_categories: List of predicted categories (Aspect Detection) for each instance
    - all_pred_sentiments: List of predicted sentiments (Sentiment Detection) for each instance
    - all_true_labels: List of true labels with aspect and sentiment information
    - num_aspect: The number of aspect labels (to split the labels correctly)

    Returns:
    - Dictionary with macro-averaged Precision, Recall, and F1-score for Aspect and Sentiment Detection
    """

    # Separate true labels into aspects and sentiments based on num_aspect
    true_acd = [label[:num_aspect] for label in all_true_labels]  # True Aspect Detection labels
    true_acsa = [label[num_aspect:] for label in all_true_labels]  # True Sentiment Detection labels

    # Flatten lists if needed (this step assumes true_acd and true_acsa are lists of lists)
    true_acd = [item for sublist in true_acd for item in sublist]
    true_acsa = [item for sublist in true_acsa for item in sublist]

    pred_acd = [item for sublist in all_pred_categories for item in sublist]
    pred_acsa = [item for sublist in all_pred_sentiments for item in sublist]

    # Calculate Precision, Recall, and F1-score for Aspect Detection
    acd_precision = precision_score(true_acd, pred_acd, average="macro", zero_division=0)
    acd_recall = recall_score(true_acd, pred_acd, average="macro", zero_division=0)
    acd_f1 = f1_score(true_acd, pred_acd, average="macro", zero_division=0)

    # Calculate Precision, Recall, and F1-score for Sentiment Detection
    acsa_precision = precision_score(true_acsa, pred_acsa, average="macro", zero_division=0)
    acsa_recall = recall_score(true_acsa, pred_acsa, average="macro", zero_division=0)
    acsa_f1 = f1_score(true_acsa, pred_acsa, average="macro", zero_division=0)

    return {
        "Aspect Detection": {
            "Precision": acd_precision * 100,
            "Recall": acd_recall * 100,
            "F1-score": acd_f1 * 100,
        },
        "Sentiment Detection": {
            "Precision": acsa_precision * 100,
            "Recall": acsa_recall * 100,
            "F1-score": acsa_f1 * 100,
        }
    }
#metrics
# print(calculate_macro_metrics(predictions[:,0:10], predictions[:,10:20], y_test))
model_path = "/Users/khangdoan/Documents/GitHub/ABSA/Model/ML_Model"
with open (model_path, 'wb') as f:
    pickle.dump(new_clf, f)
