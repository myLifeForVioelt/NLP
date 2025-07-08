import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# 配置
CHINESE_TEXT_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MSCTD\\chinese_train.txt"
CHINESE_LABEL_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MSCTD\\sentiment_train.txt"
ENGLISH_CSV_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MELD\\MELD.Raw\\train_sent_emo.csv"

BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 1e-3
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_CLASSES = 3  # positive, negative, neutral
MAX_LEN = 64

# 标签映射，保证中英文一致
label_map_ch = {0:0, 1:1, 2:2}  # 中文标签本身就是数字0,1,2，直接用
label_map_en = {'neutral': 0, 'negative': 1, 'positive': 2}

# 简单中文分词函数（按字切）
def tokenize(text):
    text = text.lower()
    # 对英文也适用简单词分割，如果有英文就按单词，不然中文按字
    if re.search(r'[a-zA-Z]', text):
        tokens = re.findall(r'\b\w+\b', text)
    else:
        tokens = list(text)
    return tokens

# 构建词表
def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for sent in sentences:
        tokens = tokenize(sent)
        counter.update(tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

# 文本转索引
def text_to_indices(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

# 自定义 Dataset
class ChineseSentimentDataset(Dataset):
    def __init__(self, text_path, label_path, vocab, max_len=64):
        with open(text_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f.readlines()]
        with open(label_path, "r", encoding="utf-8") as f:
            self.labels = [int(line.strip()) for line in f.readlines()]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = text_to_indices(self.texts[idx], self.vocab)
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        label = self.labels[idx]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class EnglishSentimentDataset(Dataset):
    def __init__(self, csv_path, vocab, max_len=64):
        self.data = pd.read_csv(csv_path)
        self.vocab = vocab
        self.max_len = max_len
        self.label_map = label_map_en

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['Utterance'])
        sentiment = row['Sentiment'].strip().lower()
        label = self.label_map.get(sentiment, 0)
        indices = text_to_indices(text, self.vocab)
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 自定义 collate_fn，实现 padding
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts_padded, labels

# GRU模型定义
class GRUEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1, bidirectional=True, dropout=0.3):
        super(GRUEmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        last_output = gru_out[:, -1, :]
        out = self.dropout(last_output)
        logits = self.fc(out)
        return logits

# 训练函数
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 测试函数
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc

# 主程序
def main():
    # 读取中文文本构建词表
    with open(CHINESE_TEXT_PATH, "r", encoding="utf-8") as f:
        chinese_texts = [line.strip() for line in f.readlines()]

    vocab = build_vocab(chinese_texts, min_freq=2)
    print(f"词表大小: {len(vocab)}")

    train_dataset = ChineseSentimentDataset(CHINESE_TEXT_PATH, CHINESE_LABEL_PATH, vocab, MAX_LEN)
    test_dataset = EnglishSentimentDataset(ENGLISH_CSV_PATH, vocab, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUEmotionClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
