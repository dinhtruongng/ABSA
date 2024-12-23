import pandas as pd
import re
import unicodedata
import torch.nn as nn
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
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ', ' okie' : ' ok',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' tích cực ',
        'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' tích cực ','hehe': ' tích cực ','hihi': ' tích cực ', 'haha': ' tích cực ', 'hjhj': ' tích cực ',
        ' lol ': ' tiêu cực ',' cc ': ' tiêu cực ','cute': u' dễ thương ','huhu': ' tiêu cực ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
        'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' tích cực ', 'store': u' cửa hàng ',
        'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
        'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',
        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
        'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
        'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
        'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',
        'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' tích cực ',
        ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback ',
        ' h ' : u' giờ', 
        ' e ' : u' em'
    }
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

def final_preprocess_v3(table):
    aspect_categories = ['BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL', 
                     'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE']
    polarity_to_idx = { 'Positive': 0, 'Negative': 1, 'Neutral': 2 }
    table['comment'] = table['comment'].apply(preprocess_comment)
    preprocess_label(table)
    table['label'] = table['label'].apply(lambda x: label_to_tensor_v3(x, aspect_categories, polarity_to_idx))
    return table