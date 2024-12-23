import pandas as pd 
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
from pyvi import ViTokenizer, ViPosTagger

with open ('/Users/khangdoan/Documents/GitHub/ABSA/Model/ML_Model', 'rb') as f:
    loaded_model = pickle.load(f)
with open ('/Users/khangdoan/Documents/GitHub/ABSA/Model/vectorizer_for_ML', 'rb') as f:
    vecto = pickle.load(f)

def lowercase(df):
        df['comment'] = df['comment'].str.lower()

def remove_punc(text):
    punc = string.punctuation
    return text.translate(str.maketrans('', '', punc))

def final_rmv_punc(df):
    df['comment'] = df['comment'].apply(remove_punc)

def remove_num(df):
    df['comment'] = df['comment'].replace(to_replace=r'\d', value='', regex=True)

def tokenize(df):
    tokenizer = ViTokenizer.tokenize
    df['comment'] = df['comment'].apply(tokenizer)

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
    
    # Substitute emojis with a space
    text = emoji_pattern.sub(r" ", text)
    return text

def final_remove_emote(df):
    df['comment'] = df['comment'].apply(remove_emote)    
    return df

def spliY(y, label):
    label = {'BATTERY':[],'CAMERA':[],'DESIGN':[],'FEATURES':[],'GENERAL':[],'OTHERS':[],'PERFORMANCE':[],'PRICE':[],'SCREEN':[],'SER&ACC':[],'STORAGE':[]}
    for i in y:
        x = i[:-1].split(";")
        # print(x)
        temp = list(label.keys())
        for j in x:
            j = j.replace('{', '')
            j = j.replace('}', '')
            t = j.split('#')
            
            if t[0] != "OTHERS":
                temp.remove(t[0])
                if t[1] == "Positive":
                    label[t[0]].append(1) #nhan xet tich cuc
                elif t[1] == "Neutral":
                    label[t[0]].append(0) #nhan xet trung binh 
                else:
                    label[t[0]].append(-1) #nhan xet tieu cuc
                    
            else:
                temp.remove(t[0])
                label['OTHERS'].append(1) #phan loai other, ko lien quan
                
        for key in temp:
                label[key].append(-2) #Ko de cap den trong phan nhan xet
    return label

def predict_input(user_input, model, vectorizer):
    # user_input = 'Máy đẹp, màn hình đẹp, pin trâu, cấu hình mạnh'
    user_input = user_input.lower()
    user_input = remove_punc(user_input)
    user_input = remove_emote(user_input)
    vector = vectorizer.transform([user_input])
    output = model.predict(vector)
    output_dict = {'BATTERY':[],'CAMERA':[],'DESIGN':[],'FEATURES':[],'GENERAL':[],'OTHERS':[],'PERFORMANCE':[],'PRICE':[],'SCREEN':[],'SER&ACC':[],'STORAGE':[]}
    output = output[0]
    index = 0
    
    for keys in output_dict.keys():
        if keys == 'OTHERS':
            if output[index] == 1:
                output_dict[keys].append("OTHERS")
            else:
                output_dict[keys].append("non")
        else:
            if output[index] == 1:
                output_dict[keys].append("Positive")
            elif output[index] == 0:
                output_dict[keys].append("Neutral")
            elif output[index] == -1:
                output_dict[keys].append("Negative")
            else:
                output_dict[keys].append("non")
        index += 1
    for key in output_dict.keys():
        if output_dict[key][0] != 'non' and output_dict[key][0] != 'OTHERS':
            print(key + ": " + output_dict[key][0])
        elif output_dict[key][0] == 'OTHERS':
            print('OTHERS')

def infer_ML_model(text):
    predict_input(text, loaded_model, vecto)
    return 0
infer_ML_model("Máy đẹp, màn hình đẹp, pin trâu, cấu hình mạnh")