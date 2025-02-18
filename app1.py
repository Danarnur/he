import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, multilabel_confusion_matrix
import streamlit as st  # Tambahkan Streamlit

# Load dataset
df = pd.read_csv("/content/cobalabel1.csv")  # Pastikan path sudah benar

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# Tokenization and Embedding Layer
class BertEmbedding(nn.Module):
    def __init__(self):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained("indobenchmark/indobert-base-p1")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        review = str(self.data.iloc[index]["Ulasan"])
        inputs = self.tokenizer(review, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")

        labels = self.data.iloc[index][["LP", "LN", "HP", "HN", "PP", "PN"]].astype(float).values
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels
        }

# Split data 80:20
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
dataset = SentimentDataset(df, tokenizer)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Fine-Tuned Model class
class IndoBERTMultiAspect(nn.Module):
    def __init__(self):
        super(IndoBERTMultiAspect, self).__init__()
        self.embedding = BertEmbedding()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 6)  # 768 adalah ukuran hidden layer BERT
    
    def forward(self, input_ids, attention_mask):
        pooled_output = self.embedding(input_ids, attention_mask)
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        return logits

# Model, loss, optimizer
model = IndoBERTMultiAspect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 5
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    print(f"Epoch {epoch+1}, Training Loss: {train_losses[-1]}")

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return all_labels, all_preds, total_loss / len(loader)

# Get evaluation results
all_labels, all_preds, test_loss = evaluate_model(model, test_loader)
val_losses.append(test_loss)

# Calculate metrics dynamically
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
f1 = f1_score(all_labels, all_preds, average='macro')
h_loss = hamming_loss(all_labels, all_preds)

# Display metrics dynamically using Streamlit
st.title("Model Evaluation Results")
st.write(f"Test Loss: {test_loss:.4f}")
st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")
st.write(f"Hamming Loss: {h_loss:.2f}")

# Plot Loss
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Confusion Matrix
conf_matrix = multilabel_confusion_matrix(all_labels, all_preds)
aspect_names = ["LP", "LN", "HP", "HN", "PP", "PN"]

for i, matrix in enumerate(conf_matrix):
    plt.figure(figsize=(5,4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Relevant', 'Relevant'], yticklabels=['Not Relevant', 'Relevant'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {aspect_names[i]}")
    plt.show()

print("Training and evaluation completed!")
