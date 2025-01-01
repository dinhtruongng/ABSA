import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define aspect and sentiment mappings
aspect2idx = {
    'CAMERA': 0, 'FEATURES': 1, 'BATTERY': 2, 'PERFORMANCE': 3,
    'DESIGN': 4, 'GENERAL': 5, 'PRICE': 6, 'SCREEN': 7, 'SER&ACC': 8, 'STORAGE': 9
}
sentiment2idx = {
    'Positive': 2, 'Neutral': 1, 'Negative': 0
}
idx2aspect = dict(zip(aspect2idx.values(), aspect2idx.keys()))
idx2sentiment = dict(zip(sentiment2idx.values(),sentiment2idx.keys()))

num_aspect = len(aspect2idx)

# Convert label cell to tensor
def convert_label(cell):
    return torch.tensor([float(x) for x in cell.strip('[]').split()])

# Load train data
train = pd.read_csv("Train_preprocessed_with_-1.csv")
sentences_train = list(train['comment'])
labels_train = list(train['label'].apply(convert_label))

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

# Define dataset
class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        dic = {key: val.squeeze(0) for key, val in encoding.items()}
        dic['labels'] = labels
        return dic

# Create dataset and dataloader
train_dataset = CustomTextDataset(sentences_train, labels_train, tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)

dev = pd.read_csv('dev_final.csv')
sentences_dev = list(dev['comment'])
labels_dev = list(dev['label'].apply(convert_label))

dev_dataset = CustomTextDataset(sentences_dev, labels_dev, tokenizer)
dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=128)

class HierachyAttention(nn.Module):
    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)  # (batch_size, seq_len, out_features)
        u = torch.tanh(u)
        similarities = self.uw(u)  # (batch_size, seq_len, 1)
        similarities = similarities.squeeze(dim=-1)  # (batch_size, seq_len)

        # Mask the similarities
        similarities = similarities.masked_fill(~mask.bool(), -float('inf'))

        if self.softmax:
            alpha = torch.softmax(similarities, dim=-1)
            return alpha
        else:
            return similarities
            # return attention score

def element_mul(input1, input2, return_not_sum_result=False):
        output = input1 * input2.unsqueeze(2)  # Ensure correct broadcasting
        result = output.sum(dim=1)
        if return_not_sum_result:
            return result, output
        else:
            return result

class JointModel(nn.Module):
    def __init__(self, word_embedder, categories, polarities):
        super().__init__()
        self.word_embedder = word_embedder
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss(ignore_index=-1)

        embed_dim = word_embedder.embedding_dim
        self.embedding_layer_fc = nn.Linear(embed_dim, embed_dim)
        self.embedding_layer_aspect_attentions = nn.ModuleList([HierachyAttention(embed_dim, embed_dim) for _ in range(self.category_num)])
        self.lstm_layer_aspect_attentions = nn.ModuleList([HierachyAttention(embed_dim, embed_dim) for _ in range(self.category_num)])

        self.lstm = nn.LSTM(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)

        self.category_fcs = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(self.category_num)])
        self.sentiment_fc = nn.Sequential(nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Linear(32, self.polarity_num))

    def forward(self, tokens, labels, mask, threshold=0.25):
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        embeddings = word_embeddings
        embedding_layer_category_outputs = []
        embedding_layer_sentiment_outputs = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(embeddings, mask)

            category_output = element_mul(embeddings, alpha)
            embedding_layer_category_outputs.append(category_output)

            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, embeddings.transpose(1, 2)).squeeze(1)
            sentiment_alpha = softmax(sentiment_alpha, dim=-1)
            sentiment_output = torch.matmul(sentiment_alpha.unsqueeze(1), word_embeddings).squeeze(1)
            embedding_layer_sentiment_outputs.append(sentiment_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)

        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(lstm_result, mask)
            category_output = element_mul(lstm_result, alpha)
            lstm_layer_category_outputs.append(category_output)

            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2)).squeeze(1)
            sentiment_alpha = softmax(sentiment_alpha, dim=-1)
            sentiment_output = torch.matmul(sentiment_alpha.unsqueeze(1), lstm_result).squeeze(1)
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = torch.cat([embedding_layer_category_outputs[i], lstm_layer_category_outputs[i]], dim=-1)
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = torch.cat([embedding_layer_sentiment_outputs[i], lstm_layer_sentiment_outputs[i]], dim=-1)
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        loss = 0
        if labels is not None:
            category_labels = labels[:, :self.category_num]
            polarity_labels = labels[:, self.category_num:]

            for i in range(self.category_num):
                category_mask = (category_labels[:, i] != -1)  # Mask out ignored labels
                sentiment_mask = (polarity_labels[:, i] != -1)

                if category_mask.any():  # Only calculate if there are valid labels
                    category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(-1)[category_mask], category_labels[:, i][category_mask])
                    loss += category_temp_loss

                if sentiment_mask.any():  # Only calculate if there are valid labels
                    sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i][sentiment_mask], polarity_labels[:, i][sentiment_mask].long())
                    loss += sentiment_temp_loss

#         output = {
#             'pred_category': [torch.sigmoid(e) for e in final_category_outputs],
#             'pred_sentiment': [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
#         }
        # formatting output
        final_category_outputs = [torch.sigmoid(e) for e in final_category_outputs]
        final_sentiment_outputs = [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
        final_sentiment_outputs = [torch.argmax(e, dim=-1) for e in final_sentiment_outputs]
        
        final_categories = []
        final_sentiments = []

        for i in range(len(final_category_outputs)):
            batch_category = []
            batch_sentiment = []
            for j, category_score in enumerate(final_category_outputs[i]):
                # Apply threshold for aspect detection
                if category_score >= threshold:
                    batch_category.append(1)  # Aspect detected
                    batch_sentiment.append(final_sentiment_outputs[i][j].item())
                else:
                    batch_category.append(0)  # Aspect not detected
                    batch_sentiment.append(-1)  # Set sentiment to -1 for undetected aspect
            final_categories.append(batch_category)
            final_sentiments.append(batch_sentiment)
        final_categories = torch.tensor(final_categories)
        final_sentiments = torch.tensor(final_sentiments)
        
        output = {
            'pred_category': torch.transpose(final_categories, 0, 1), # batch_size*10
            'pred_sentiment': torch.transpose(final_sentiments, 0, 1) # batch_size*10
        }

        return output, loss

w2v = 'W2V_150.txt'
embedding_dim = 150
word_to_vec = {}
with open(w2v, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        word_to_vec[word] = vector

vocab = tokenizer.get_vocab()
vocab_size = len(vocab)
E = np.zeros((vocab_size, embedding_dim))
for word, idx in vocab.items():
    E[idx] = word_to_vec.get(word, np.random.normal(scale=0.6, size=(embedding_dim,)))

embedding_matrix = torch.tensor(E, dtype=torch.float32)
embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

categories = aspect2idx.keys()
polarities = sentiment2idx.keys()
model = JointModel(embedding_layer, categories, polarities)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
model.to(device)

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

def eval_dev(model, dev_dataloader):
    model.eval()
    
    with torch.inference_mode():
        loss=0
        pred_cate = []
        pred_sent = []
        true_label = []
        
        for batch in dev_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            att_mask = batch['attention_mask'].to(device)
            output, loss = model(input_ids, labels, att_mask)
            l2_reg = torch.tensor(0., requires_grad=False)  # Initialize L2 regularization term
            for param in model.parameters():
#                 if param not in model.lstm.parameters():
                l2_reg = l2_reg + torch.norm(param, 2)  # Add squared norm of each parameter

            # Add L2 regularization term to loss
            loss = loss + l2_lambda * l2_reg
            
            pred_cate.append(output['pred_category'])
            pred_sent.append(output['pred_sentiment'])
            true_label.append(batch['labels'])
    categories = torch.cat(pred_cate, dim=0)
    sentiments = torch.cat(pred_sent, dim=0)
    labels = torch.cat(true_label, dim=0)
    metric = calculate_macro_metrics(categories, sentiments, labels)
    f1_acd = metric['Aspect Detection']['F1-score']
    f1_sc = metric['Sentiment Detection']['F1-score']
    
    loss = loss / len(dev_dataloader)
    return loss, f1_acd, f1_sc

# Training loop
epochs = 50
l2_lambda = 0.01
for epoch in range(epochs):
    model.train()
    total_loss = 0
    pred_cate = []
    pred_sent = []
    true_label = []
    
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        att_mask = batch['attention_mask'].to(device)
        output, loss = model(input_ids, labels, att_mask)
#         output, loss = model(batch['input_ids'], batch['labels'], batch['attention_mask'])
        l2_reg = torch.tensor(0., requires_grad=True)  # Initialize L2 regularization term
        for param in model.parameters():
#             if param not in model.lstm.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)  # Add squared norm of each parameter

        # Add L2 regularization term to loss
        loss = loss + l2_lambda * l2_reg
        
        pred_cate.append(output['pred_category'])
        pred_sent.append(output['pred_sentiment'])
        true_label.append(batch['labels'])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
        
        
    # metric
    categories = torch.cat(pred_cate, dim=0)
    sentiments = torch.cat(pred_sent, dim=0)
    labels = torch.cat(true_label, dim=0)
#     print(labels.shape)
#     print(categories.shape)
#     print(sentiment.shape)
    metric = calculate_macro_metrics(categories, sentiments, labels)
    f1_acd = metric['Aspect Detection']['F1-score']
    f1_sc = metric['Sentiment Detection']['F1-score']
    dev = eval_dev(model, dev_dataloader)
    
    print(f"Epoch {epoch + 1} - Train_Loss: {total_loss / len(train_dataloader)} - train_acd_f1: {f1_acd} - train_sc_f1: {f1_sc}")
    print(f"dev_loss: {dev[0]} - dev_acd_f1: {dev[1]} - dev_sc_f1: {dev[2]}")
    
    if epoch%10==0 and epoch>0:
        torch.save(model.state_dict(), f"JointModel_checkpoint{epoch}.pth")
