import pandas as pd
import re
import unicodedata
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from torchtext.vocab import FastText
import math

#data
train = pd.read_csv('Data/Train.csv')
test = pd.read_csv('Data/Test.csv')

#preprocess
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

#labels are tensors of size 20 cat of [aspect, polarity]
def label_to_tensor_v3(label: str, aspect_categories: list, polarity_to_idx: dict):
    aspect_categories = ['BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL', 
                         'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE']
    polarity_to_idx = { 'Positive': 0, 'Negative': 1, 'Neutral': 2 }
    aspect_tensor = torch.zeros(len(aspect_categories), dtype=torch.float32)  
    polarity_tensor = torch.full((len(aspect_categories),), -1, dtype=torch.float32)  
    components = label.split(';')
    for component in components:
        component = component.strip('{}')
        if '#' not in component:
            continue
        
        aspect, polarity = component.split('#')
        if aspect in aspect_categories and polarity in polarity_to_idx:
            aspect_idx = aspect_categories.index(aspect)
            aspect_tensor[aspect_idx] = 1  
            polarity_tensor[aspect_idx] = polarity_to_idx[polarity] 

    full_tensor = torch.cat([aspect_tensor, polarity_tensor])
    return full_tensor
 
def final_preprocess_v3(table):
    aspect_categories = ['BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL', 
                     'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE']
    polarity_to_idx = { 'Positive': 0, 'Negative': 1, 'Neutral': 2 }
    table['comment'] = table['comment'].apply(preprocess_comment)
    preprocess_label(table)
    table['label'] = table['label'].apply(lambda x: label_to_tensor_v3(x, aspect_categories, polarity_to_idx))
    return table

train = final_preprocess_v3(train)
test = final_preprocess_v3(test)
#tokenize
tokenizer = AutoTokenizer.from_pretrained("bkai-foundation-models/vietnamese-bi-encoder")
fastText = FastText(language='vi')

MAX_LENGTH = 128
BATCH_SIZE = 32
EMBEDDING_DIM = fastText.dim  
VOCAB_SIZE = tokenizer.vocab_size

encoding = tokenizer(
    train['comment'].tolist(),
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)
token_ids = encoding['input_ids']
tokens = []

for each in token_ids:
    temp = tokenizer.convert_ids_to_tokens(each)
    tokens.append(temp)

embedding_matrix = torch.zeros(VOCAB_SIZE, EMBEDDING_DIM)
for each in tokens:
    for token in each:
        if token in fastText.stoi:
            vector=torch.tensor(fastText[token], dtype=torch.float32)
        else:
            vector = torch.zeros(EMBEDDING_DIM)
        embedding_matrix[tokenizer.convert_tokens_to_ids(token)] = vector

def tokenize(df):
    texts = df["comment"].tolist()
    labels = df["label"].tolist()
    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    token_ids = encoding['input_ids'] 
    return token_ids, labels


class TextDataset(Dataset):
    def __init__(self, token_ids, labels):
        self.token_ids = token_ids
        self.labels = labels
    def __len__(self):
        return len(self.token_ids)
    def __getitem__(self, idx):
        tokens = self.token_ids[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tokens, label

def create_dataloader(df, batch_size=BATCH_SIZE):
    token_ids, labels = tokenize(df)
    dataset = TextDataset(token_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

train_loader = create_dataloader(train)
test_loader = create_dataloader(test)

#model
class Attention(nn.Module):
    def __init__(self, embed_dim=EMBEDDING_DIM, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        if score_function == 'nl':
            self.weight = nn.parameter(torch.Tensor(hidden_dim, hidden_dim))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2: 
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        # in sentiment analysis, they focus to the importance of k, so maybe we dont have V  value (intuitively, V is k...)
        output = torch.bmm(score, kx) 
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  
        output = self.proj(output)  
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''
    def __init__(self, embed_dim=EMBEDDING_DIM, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)

class ATAE_LSTM(nn.Module):
    def __init__(self, embed_matrix, hidden_dim=128, embedding_dim=EMBEDDING_DIM, polarities_dim=3, num_aspects=10):
        super(ATAE_LSTM, self).__init__()
        
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embed_matrix, dtype=torch.float))
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_aspects = num_aspects
        self.polarities_dim = polarities_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(hidden_dim, score_function='bi_linear')

        self.category_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(num_aspects)
        ])

        self.sentiment_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, polarities_dim)
            ) for _ in range(num_aspects)
        ])

        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, text_indices, labels=None, threshold=0.25):
        x = self.embed(text_indices)
        h, _ = self.lstm(x)
        _, score = self.attention(h)
        pooled_output = torch.bmm(score, h).squeeze(dim=1)  # Shape: (batch_size, hidden_dim)

        final_category_outputs = []
        final_sentiment_outputs = []

        for i in range(self.num_aspects):
            # Category and sentiment predictions
            category_output = self.category_fcs[i](pooled_output)
            sentiment_output = self.sentiment_fc[i](pooled_output)
            final_category_outputs.append(category_output)
            final_sentiment_outputs.append(sentiment_output)

        loss = 0
        if labels is not None:
            category_labels = labels[:, :self.num_aspects]
            polarity_labels = labels[:, self.num_aspects:]

            for i in range(self.num_aspects):
                if polarity_labels.size(1) <= i:  
                    continue

                category_mask = (category_labels[:, i] != -1)  
                sentiment_mask = (polarity_labels[:, i] != -1)

                if category_mask.any():  
                    category_temp_loss = self.category_loss(
                        final_category_outputs[i].squeeze(-1)[category_mask],
                        category_labels[:, i][category_mask]
                    )
                    loss += category_temp_loss

                if sentiment_mask.any():  
                    sentiment_temp_loss = self.sentiment_loss(
                        final_sentiment_outputs[i][sentiment_mask],
                        polarity_labels[:, i][sentiment_mask].long()
                    )
                    loss += sentiment_temp_loss

        final_category_outputs = [torch.sigmoid(e) for e in final_category_outputs]
        final_sentiment_outputs = [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
        
        
        final_categories = []
        final_sentiments = []

        for i in range(len(final_category_outputs)):
            batch_category = []
            batch_sentiment = []
            for j, category_score in enumerate(final_category_outputs[i]):
                if category_score >= threshold:
                    batch_category.append(1)  
                    batch_sentiment.append(torch.argmax(final_sentiment_outputs[i][j]).item())
                else:
                    batch_category.append(0)  
                    batch_sentiment.append(-1)  
            final_categories.append(batch_category)
            final_sentiments.append(batch_sentiment)

        final_categories = torch.tensor(final_categories)
        final_sentiments = torch.tensor(final_sentiments)

        output = {
            'pred_category': torch.transpose(final_categories, 0, 1),  # (batch_size, num_aspects)
            'pred_sentiment': torch.transpose(final_sentiments, 0, 1)  # (batch_size, num_aspects)
        }

        return output, loss

#train

model_v2 = ATAE_LSTM(embed_matrix=embedding_matrix, hidden_dim=128, embedding_dim=EMBEDDING_DIM, polarities_dim=3, num_aspects=10)
optimizer_v2 = torch.optim.Adam(model_v2.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_v2.to(device)

EPOCHS = 20
model_v2.train()

for epoch in range(EPOCHS):
    correct_categories, correct_sentiments, total_categories, total_sentiments = 0, 0, 0, 0
    total_loss = 0.0

    for i, batch in enumerate(train_loader):
        tokens, labels = batch
        tokens, labels = tokens.to(device), labels.to(device)

        optimizer_v2.zero_grad()

        output, loss = model_v2(tokens, labels)
        total_loss += loss.item()

        pred_categories = output['pred_category'].to(device)  # Shape: (batch_size, num_aspects)
        pred_sentiments = output['pred_sentiment'].to(device)  # Shape: (batch_size, num_aspects)

        valid_category_mask = labels[:, :model_v2.num_aspects] != -1
        valid_sentiment_mask = labels[:, model_v2.num_aspects:] != -1
        
        correct_categories += (pred_categories[valid_category_mask] == labels[:, :model_v2.num_aspects][valid_category_mask]).sum().item()
        correct_sentiments += (pred_sentiments[valid_sentiment_mask] == labels[:, model_v2.num_aspects:][valid_sentiment_mask]).sum().item()

        total_categories += valid_category_mask.sum().item()
        total_sentiments += valid_sentiment_mask.sum().item()

        loss.backward()
        optimizer_v2.step()

    avg_loss = total_loss / len(train_loader)
    category_accuracy = correct_categories / total_categories if total_categories > 0 else 0.0
    sentiment_accuracy = correct_sentiments / total_sentiments if total_sentiments > 0 else 0.0

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Category Accuracy: {category_accuracy:.4f}, Sentiment Accuracy: {sentiment_accuracy:.4f}")

# Save the model
torch.save(model_v2.state_dict(), 'UI/ATAE_checkpoint20.pth')

from sklearn.metrics import precision_score, recall_score, f1_score

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

def eval_dev(model, dev_dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    
    with torch.inference_mode():
        pred_cate = []
        pred_sent = []
        true_label = []
        
        for batch in dev_dataloader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            output, _ = model(input_ids, labels)
            
            pred_cate.append(output["pred_category"].cpu())
            pred_sent.append(output["pred_sentiment"].cpu())
            true_label.append(labels.cpu())
    
    categories = torch.cat(pred_cate, dim=0).tolist()
    sentiments = torch.cat(pred_sent, dim=0).tolist()
    labels = torch.cat(true_label, dim=0).tolist()
    
    metric = calculate_macro_metrics(categories, sentiments, labels)
    f1_acd = metric['Aspect Detection']['F1-score']
    f1_sc = metric['Sentiment Detection']['F1-score']
    return f1_acd, f1_sc


f1_acd, f1_sc = eval_dev(model_v2, test_loader)
print(f"Aspect Detection F1-score: {f1_acd:.2f}")
print(f"Sentiment Detection F1-score: {f1_sc:.2f}")