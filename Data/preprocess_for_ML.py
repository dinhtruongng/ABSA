import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
from pyvi import ViTokenizer
train_df = pd.read_csv("/Users/khangdoan/Documents/GitHub/ABSA/Data/Train.csv")
test_df = pd.read_csv("/Users/khangdoan/Documents/GitHub/ABSA/Data/Test.csv")
FULL = pd.concat([train_df, test_df], ignore_index=True)
def lowercase(df):
    df['comment'] = df['comment'].str.lower()

lowercase(FULL)
import string
punc = string.punctuation
def remove_punc(text):
    return text.translate(str.maketrans('', '', punc))
def final_rmv_punc(df):
    df['comment'] = df['comment'].apply(remove_punc)

final_rmv_punc(FULL)
def remove_num(df):
    df['comment'] = df['comment'].replace(to_replace=r'\d', value='', regex=True)

remove_num(FULL)

tokenizer = ViTokenizer.tokenize
def tokenize(df):
    df['comment'] = df['comment'].apply(tokenizer)
tokenize(FULL)
def remove_emote(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"  # other symbols
        u"\U000024C2-\U0001F251"  # enclosed characters
        u"\U0001f926-\U0001f937"  # gestures
        u"\U0001F1F2"             # specific characters
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"          # gender symbols
        "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r" ", text)
    return text
def final_remove_emote(df):
    df['comment'] = df['comment'].apply(remove_emote)    
    return df
FULL = final_remove_emote(FULL)
X = FULL['comment']
y = FULL.iloc[:, -1]
label = {'BATTERY':[],'CAMERA':[],'DESIGN':[],'FEATURES':[],'GENERAL':[],'PERFORMANCE':[],'PRICE':[],'SCREEN':[],'SER&ACC':[],'STORAGE':[]}
def spliY(y, label):
    label = {'BATTERY_avai':[],'CAMERA_avai':[],'DESIGN_avai':[],'FEATURES_avai':[],'GENERAL_avai':[],'PERFORMANCE_avai':[],'PRICE_avai':[],'SCREEN_avai':[],'SER&ACC_avai':[],'STORAGE_avai':[],'BATTERY':[],'CAMERA':[],'DESIGN':[],'FEATURES':[],'GENERAL':[],'PERFORMANCE':[],'PRICE':[],'SCREEN':[],'SER&ACC':[],'STORAGE':[]}
    
    for i in y:
        x = i[:-1].split(";")
        temp = ['BATTERY','CAMERA','DESIGN','FEATURES','GENERAL','PERFORMANCE','PRICE','SCREEN','SER&ACC','STORAGE']
        for j in x:
            j = j.replace('{', '')
            j = j.replace('}', '')
            t = j.split('#')
            
            if t[0] != "OTHERS":
                temp.remove(t[0])
                if t[1] == "Positive":
                    label[t[0]].append(2) #nhan xet tich cuc                    
                    label[str(t[0]) + "_avai"].append(1)
                elif t[1] == "Neutral":
                    label[t[0]].append(0) #nhan xet trung binh 
                    label[str(t[0]) + "_avai"].append(1)
                else:
                    label[t[0]].append(1) #nhan xet tieu cuc
                    label[str(t[0]) + "_avai"].append(1)
        for key in temp:
            label[key].append(-1) #Ko de cap den trong phan nhan xet
            label[str(key) + "_avai"].append(0)
    return label
y = spliY(y, label)
y = pd.DataFrame(y)

cleanedData = []
for sentence in X:
    cleanedData.append(sentence)
vectorizer = CountVectorizer(max_features=10000)
BoW = vectorizer.fit_transform(cleanedData)
vecto_path = "/Users/khangdoan/Documents/GitHub/ABSA/Model/vectorizer_for_ML"
BoW_path = "/Users/khangdoan/Documents/GitHub/ABSA/Model/BoW_for_ML"
y_path = "/Users/khangdoan/Documents/GitHub/ABSA/Model/y"

with open (vecto_path, 'wb') as f:
    pickle.dump(vectorizer, f)
with open (BoW_path, 'wb') as f:
    pickle.dump(BoW, f)
with open (y_path, 'wb') as f:
    pickle.dump(np.asarray(y), f)


# x_train,x_test,y_train,y_test = train_test_split(BoW,np.asarray(y))

# param = {
#     'estimator__C': [0.1, 1, 10, 100],
#     'estimator__gamma': [1, 0.1, 0.01, 0.001],
#     'estimator__kernel': ['rbf', 'linear']
# }

# model = MultiOutputClassifier(SVC())

# grid = GridSearchCV(model, param, cv = 2) 

# clf = grid.fit(x_train, y_train)
# res = pd.DataFrame(grid.cv_results_)
# best_para = clf.best_params_
# new_clf = MultiOutputClassifier(SVC(C=10, gamma=0.01, kernel='rbf')).fit(x_train, y_train)
# predictions = new_clf.predict(x_test)
# with open ('vecto1', 'wb') as f:
#     pickle.dump(vectorizer, f)
# with open ('model_pickle1', 'wb') as f:
#     pickle.dump(new_clf, f)