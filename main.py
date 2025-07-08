import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# ---------- 配置 ----------
CHINESE_TEXT_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MSCTD\\chinese_train.txt"
CHINESE_LABEL_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MSCTD\\sentiment_train.txt"
ENGLISH_CSV_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MELD\\MELD.Raw\\train_sent_emo.csv"

PRETRAINED_MODEL = "bert-base-multilingual-uncased"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 64
NUM_CLASSES = 3  # positive, negative, neutral

# ---------- 中文训练数据集 ----------
class ChineseSentimentDataset(Dataset):
    def __init__(self, text_path, label_path, tokenizer, max_len=64, culture_id=0):
        with open(text_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f.readlines()]
        with open(label_path, "r", encoding="utf-8") as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.culture_id = culture_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return input_ids, attention_mask, torch.tensor(self.culture_id), torch.tensor(label)

# ---------- 英文测试数据集 ----------
class EnglishSentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=64, culture_id=1):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.culture_id = culture_id

        # 标签映射：与中文一致
        self.label_map = {
            'neutral': 0,
            'negative': 1,
            'positive': 2
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['Utterance'])
        sentiment = row['Sentiment'].strip().lower()
        label = self.label_map.get(sentiment, 0)  # 默认为 neutral

        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return input_ids, attention_mask, torch.tensor(self.culture_id), torch.tensor(label)

# ---------- 模型定义 ----------
class CultureAwareTextEmotionModel(nn.Module):
    def __init__(self, pretrained_model=PRETRAINED_MODEL, culture_emb_dim=8, hidden_dim=256, num_classes=3):
        super(CultureAwareTextEmotionModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.culture_embedding = nn.Embedding(2, culture_emb_dim)
        self.fc = nn.Linear(self.bert.config.hidden_size + culture_emb_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, culture_id):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = bert_output.last_hidden_state[:, 0, :]  # CLS token
        culture_feat = self.culture_embedding(culture_id)
        fused = torch.cat([text_feat, culture_feat], dim=1)
        x = F.relu(self.fc(fused))
        logits = self.out(x)
        return logits

# ---------- 训练函数 ----------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, culture_id, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        culture_id = culture_id.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, culture_id)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# ---------- 测试函数 ----------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, culture_id, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            culture_id = culture_id.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask, culture_id)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc

# ---------- 主程序 ----------
def main():
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

    # 数据集加载
    train_dataset = ChineseSentimentDataset(CHINESE_TEXT_PATH, CHINESE_LABEL_PATH, tokenizer)
    test_dataset = EnglishSentimentDataset(ENGLISH_CSV_PATH, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CultureAwareTextEmotionModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Test Acc (EN): {test_acc:.4f}")

if __name__ == "__main__":
    main()
