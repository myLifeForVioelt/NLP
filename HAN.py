import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

# ---------- 配置 ----------
CHINESE_TEXT_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MSCTD\\chinese_train.txt"
CHINESE_LABEL_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MSCTD\\sentiment_train.txt"
ENGLISH_CSV_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MELD\\MELD.Raw\\train_sent_emo.csv"

PRETRAINED_MODEL = "bert-base-multilingual-uncased"
NUM_EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-3
MAX_LEN = 64
NUM_CLASSES = 3  # positive, negative, neutral
EMBEDDING_DIM = 128
HIDDEN_SIZE = 64

# ---------- 情绪标签映射 ----------
label_map = {
    'neutral': 0,
    'negative': 1,
    'positive': 2
}

# ---------- 分词器 ----------
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
word2idx = tokenizer.get_vocab()
pad_idx = word2idx["[PAD]"]

# ---------- 中文训练数据集 ----------
class ChineseSentimentDataset(Dataset):
    def __init__(self, text_path, label_path, max_len=64, culture_id=0):
        with open(text_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f.readlines()]
        with open(label_path, "r", encoding="utf-8") as f:
            self.labels = [int(line.strip()) for line in f.readlines()]
        self.max_len = max_len
        self.culture_id = culture_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenizer.encode(self.texts[idx], max_length=self.max_len,
                                  truncation=True, padding='max_length')
        return torch.tensor(tokens), torch.tensor(self.culture_id), torch.tensor(self.labels[idx])

# ---------- 英文测试数据集 ----------
class EnglishSentimentDataset(Dataset):
    def __init__(self, csv_path, max_len=64, culture_id=1):
        self.data = pd.read_csv(csv_path)
        self.max_len = max_len
        self.culture_id = culture_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['Utterance'])
        sentiment = self.data.iloc[idx]['Sentiment'].strip().lower()
        label = label_map.get(sentiment, 0)
        tokens = tokenizer.encode(text, max_length=self.max_len, truncation=True, padding='max_length')
        return torch.tensor(tokens), torch.tensor(self.culture_id), torch.tensor(label)

# ---------- collate_fn ----------
def collate_fn(batch):
    inputs, cultures, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(cultures), torch.tensor(labels)

# ---------- Attention Layer ----------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = self.att(x).squeeze(-1)  # [batch, seq]
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)

# ---------- HAN 模型（简化版）----------
class HANEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(HANEmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len, embed]
        output, _ = self.lstm(embedded)
        attn_output = self.attention(output)
        logits = self.fc(attn_output)
        return logits

# ---------- 训练 ----------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, _, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ---------- 评估 ----------
def evaluate_model(model, dataloader, device):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for inputs, _, labels in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            labels_all.extend(labels.numpy())
    return accuracy_score(labels_all, preds)

# ---------- 主函数 ----------
def main():
    train_dataset = ChineseSentimentDataset(CHINESE_TEXT_PATH, CHINESE_LABEL_PATH)
    test_dataset = EnglishSentimentDataset(ENGLISH_CSV_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HANEmotionClassifier(len(word2idx), EMBEDDING_DIM, HIDDEN_SIZE, NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - English Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
