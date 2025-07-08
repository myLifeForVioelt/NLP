import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer

# ---------- 配置 ----------
CHINESE_TEXT_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MSCTD\\chinese_train.txt"
CHINESE_LABEL_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MSCTD\\sentiment_train.txt"
ENGLISH_CSV_PATH = "C:\\Users\\29094\\Desktop\\计算语言学\\MELD\\MELD.Raw\\train_sent_emo.csv"

PRETRAINED_MODEL = "bert-base-multilingual-uncased"
NUM_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
MAX_LEN = 64
NUM_CLASSES = 3
EMBEDDING_DIM = 128

# ---------- 标签映射 ----------
label_map = {
    'neutral': 0,
    'negative': 1,
    'positive': 2
}

# ---------- Tokenizer & Vocabulary ----------
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
word2idx = tokenizer.get_vocab()
pad_idx = word2idx["[PAD]"]

# ---------- Dataset 定义 ----------
class ChineseSentimentDataset(Dataset):
    def __init__(self, text_path, label_path, max_len=64):
        with open(text_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f.readlines()]
        with open(label_path, "r", encoding="utf-8") as f:
            self.labels = [int(line.strip()) for line in f.readlines()]
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenizer.encode(self.texts[idx], max_length=self.max_len,
                                  truncation=True, padding='max_length')
        return torch.tensor(tokens), torch.tensor(self.labels[idx])

class EnglishSentimentDataset(Dataset):
    def __init__(self, csv_path, max_len=64):
        self.data = pd.read_csv(csv_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['Utterance'])
        sentiment = self.data.iloc[idx]['Sentiment'].strip().lower()
        label = label_map.get(sentiment, 0)
        tokens = tokenizer.encode(text, max_length=self.max_len,
                                  truncation=True, padding='max_length')
        return torch.tensor(tokens), torch.tensor(label)

# ---------- Collate_fn ----------
def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels)

# ---------- TextCNN 模型 ----------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [B, L, D]
        x = x.unsqueeze(1)     # [B, 1, L, D]
        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [B, F, L-k+1]
        pooled = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]  # [B, F]
        cat = torch.cat(pooled, dim=1)  # [B, F*len(K)]
        out = self.dropout(cat)
        return self.fc(out)

# ---------- Train & Eval ----------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)

# ---------- 主函数 ----------
def main():
    train_dataset = ChineseSentimentDataset(CHINESE_TEXT_PATH, CHINESE_LABEL_PATH)
    test_dataset = EnglishSentimentDataset(ENGLISH_CSV_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(len(word2idx), EMBEDDING_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - English Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
