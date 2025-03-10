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

def predict_input(user_input, model, vectorizer):
    # user_input = 'Máy đẹp, màn hình đẹp, pin trâu, cấu hình mạnh'
    user_input = user_input.lower()
    user_input = remove_punc(user_input)
    user_input = remove_emote(user_input)
    vector = vectorizer.transform([user_input])
    output = model.predict(vector)
    output_list = [['BATTERY'],['CAMERA'],['DESIGN'],['FEATURES'],['GENERAL'],['PERFORMANCE'],['PRICE'],['SCREEN'],['SER&ACC'],['STORAGE']]
    output = output[0]
    print(output)
    for i in range(len(output_list)):
        if output[i+10] == 2:
            print(output_list[i][0], "positive")
        elif output[i+10] == 0:
            print(output_list[i][0], "neutral")
        elif output[i+10] == 1:
            print(output_list[i][0], "negative")

def infer_ML_model(text):
    predict_input(text, loaded_model, vecto)
    return 0
infer_ML_model("dùng ok mượt k nóng đt có triệu mà mở max cấu hình ròi mà không nóng, nhân_viên nhiệt_tình")