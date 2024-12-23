import pandas as pd
import re
import unicodedata
import torch
import pandas as pd
import torch

# preprocess comment
def basic(text):
    text = text.lower()
    text = re.sub(r'[^\w\s!?]', '', text)
    return text

def handle_slang(text):
    slang_dict = {
        'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ', ' okay': ' ok ', 'okê': ' ok ', ' okie' : ' ok',
        ' tks ': ' cám ơn ', 'thks': ' cám ơn ', 'thanks': ' cám ơn ', 'ths': ' cám ơn ', 'thank': ' cám ơn ',
         'kg ': ' không ', 'not': ' không ', ' kg ': ' không ', '"k ': ' không ', ' kh ': ' không ', 'kô': ' không ', 'hok': ' không ', ' kp ': ' không phải ', ' kô ': ' không ', '"ko ': ' không ', ' ko ': ' không ', ' k ': ' không ', 'khong': ' không ', ' hok ': ' không ',
        'he he': ' tích cực ', 'hehe': ' tích cực ', 'hihi': ' tích cực ', 'haha': ' tích cực ', 'hjhj': ' tích cực ',
        ' lol ': ' tiêu cực ', ' cc ': ' tiêu cực ', 'cute': ' dễ thương ', 'huhu': ' tiêu cực ', ' vs ': ' với ', 'wa': ' quá ', 'wá': ' quá', 'j': ' gì ', '“': ' ',
        ' sz ': ' cỡ ', 'size': ' cỡ ', ' đx ': ' được ', 'dk': ' được ', 'dc': ' được ', 'đk': ' được ',
        'đc': ' được ', 'authentic': ' chuẩn chính hãng ', ' aut ': ' chuẩn chính hãng ', ' auth ': ' chuẩn chính hãng ', 'thick': ' tích cực ', 'store': ' cửa hàng ',
        'shop': ' cửa hàng ', 'sp': ' sản phẩm ', 'gud': ' tốt ', 'god': ' tốt ', 'wel done': ' tốt ', 'good': ' tốt ', 'gút': ' tốt ',
        'sấu': ' xấu ', 'gut': ' tốt ', ' tot ': ' tốt ', ' nice ': ' tốt ', 'perfect': 'rất tốt', 'bt': ' bình thường ',
        'time': ' thời gian ', 'qá': ' quá ', ' ship ': ' giao hàng ', ' m ': ' mình ', ' mik ': ' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng', 'chat': ' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ', 'fresh': ' tươi ', 'sad': ' tệ ',
        'date': ' hạn sử dụng ', 'hsd': ' hạn sử dụng ', 'quickly': ' nhanh ', 'quick': ' nhanh ', 'fast': ' nhanh ', 'delivery': ' giao hàng ', ' síp ': ' giao hàng ',
        'beautiful': ' đẹp tuyệt vời ', ' tl ': ' trả lời ', ' r ': ' rồi ', ' shopE ': ' cửa hàng ', ' order ': ' đặt hàng ',
        'chất lg': ' chất lượng ', ' sd ': ' sử dụng ', ' dt ': ' điện thoại ', ' nt ': ' nhắn tin ', ' tl ': ' trả lời ', ' sài ': ' xài ', 'bjo': ' bao giờ ',
        'thik': ' thích ', ' sop ': ' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': ' rất ', 'quả ng ': ' quảng ',
        'dep': ' đẹp ', ' xau ': ' xấu ', 'delicious': ' ngon ', 'hàg': ' hàng ', 'qủa': ' quả ',
        'iu': ' yêu ', 'fake': ' giả mạo ', 'trl': 'trả lời', '><': ' tích cực ',
        ' por ': ' tệ ', ' poor ': ' tệ ', 'ib': ' nhắn tin ', 'rep': ' trả lời ', 'fback': ' feedback ', 'fedback': ' feedback ',
        ' h ' : ' giờ', ' e ' : ' em'}

    for slang, formal in slang_dict.items():
        text = re.sub(r'\b' + slang + r'\b', formal, text)
    return text

def handle_emoji(text):
    return ''.join(char for char in text if not unicodedata.category(char).startswith('So'))

def preprocess_comment(text):
    # Convert to Unicode NFC format
    text = unicodedata.normalize('NFC', text)
    text = basic(text)
    text = handle_slang(text)
    text = handle_emoji(text)
    return text

def preprocess_label (df):
    columns_to_drop = ['n_star', 'date_time']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    df['label'] = df['label'].str.replace(r';?\{OTHERS\};?', '', regex=True).str.strip(';')
    return df

def label_to_tensor_v3(label: str, aspect_categories: list, polarity_to_idx: dict):
    aspect_categories = ['BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL', 
                     'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE']
    polarity_to_idx = { 'Positive': 0, 'Negative': 1, 'Neutral': 2 }  
    tensor = torch.full((len(aspect_categories),), -1, dtype=torch.float32)
    components = label.split(';')
    
    for component in components:
        component = component.strip('{}') 
        # Skip invalid components like "OTHERS"
        if '#' not in component:
            continue
        aspect, polarity = component.split('#')
        if aspect in aspect_categories and polarity in polarity_to_idx:
            aspect_idx = aspect_categories.index(aspect)
            tensor[aspect_idx] = polarity_to_idx[polarity]  
    return tensor  

def final_preprocess_v1(table):
    aspect_categories = ['BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL', 
                     'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE']
    polarity_to_idx = { 'Positive': 0, 'Negative': 1, 'Neutral': 2 }
    table['comment'] = table['comment'].apply(preprocess_comment)
    preprocess_label(table)
    table['label'] = table['label'].apply(lambda x: label_to_tensor_v3(x, aspect_categories, polarity_to_idx))
    return table
train_df = pd.read_csv('Data/Train.csv')
test_df = pd.read_csv('Data/Test.csv')

final_preprocess_v1(train_df)
final_preprocess_v1(test_df)

train_df.to_csv('Data/Train_final_lstm.csv')
test_df.to_csv('Data/Test_final_lstm.csv')

print(train_df.head())