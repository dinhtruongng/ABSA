import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from torchtext.vocab import FastText
import math

#data
train = pd.read_csv('Data/Train_final_lstm.csv')
test = pd.read_csv('Data/Test_final_lstm.csv')

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
    def __init__(self, embed_matrix=embedding_matrix, hidden_dim=128, embedding_dim=EMBEDDING_DIM, polarities_dim=3, num_aspects=10):
        super(ATAE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embed_matrix, dtype=torch.float))
        self.num_aspects = num_aspects
        self.polarities_dim = polarities_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(hidden_dim, polarities_dim*num_aspects)

    def forward(self, text_indices):
        x = self.embed(text_indices)
        h, _ = self.lstm(x)
        _, score = self.attention(h)
        output = torch.bmm(score, h).squeeze(dim=1)  # Squeeze to (batch_size, hidden_dim)
        out = self.dense(output)
        out = out.view(-1, self.num_aspects, self.polarities_dim)
        return out

#train
model_v2 = ATAE_LSTM()
criterion_v2 = nn.CrossEntropyLoss(ignore_index=-1) 
optimizer_v2 = torch.optim.Adam(model_v2.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_v2.to(device)
criterion_v2.to(device)

EPOCHS = 20
model_v2.train()
for epoch in range(EPOCHS):
    correct, total = 0, 0
    total_loss = 0.0

    for i, batch in enumerate(train_loader):
        tokens, labels = batch
        tokens, labels = tokens.to(device), labels.to(device)
        labels = labels.long()  
        optimizer_v2.zero_grad()
        output = model_v2(tokens)  # [batch_size, num_aspects,polarities_dim]
        output = output.view(-1, model_v2.polarities_dim)  # [batch_size*num_aspects, polarities_dim]
        labels = labels.view(-1)  # [batch_size *num_aspects]
        loss_v2 = criterion_v2(output, labels)
        total_loss += loss_v2.item()
        loss_v2.backward()
        optimizer_v2.step()
        predictions = torch.argmax(output, dim=-1)  # [batch_size * num_aspects]
        valid_mask = labels != -1  
        correct += (predictions[valid_mask] == labels[valid_mask]).sum().item()
        total += valid_mask.sum().item()

    avg_loss = total_loss / len(train_loader) 
    avg_acc = correct / total if total > 0 else 0.0  

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

torch.save(model_v2.state_dict(), 'Data/ATAE_checkpoint20.pth')