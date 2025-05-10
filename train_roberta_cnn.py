import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

class RobertaCNNClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model_name="roberta-base", dropout=0.3):
        super(RobertaCNNClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        x = roberta_out.last_hidden_state  #[batch_size, seq_len, hidden_dim]
        x = x.permute(0, 2, 1)             # ->[batch_size, hidden_dim, seq_len]
        x = self.relu(self.conv1(x))       # ->[btach_size, 256, seq_len]
        x = self.pool(x).squeeze(2)        # ->[batch_size, 256]
        x = self.dropout(x)
        logits = self.fc(x)                # ->[batch_size, num_classes]
        return logits
    
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
#  ====== training function ======
def train_model(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

from preprocessing import load_and_clean_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.optim import Adam

posts_df, comments_df = load_and_clean_data("data/facebook posts.xlsx", "data/facebook comments.xlsx")
texts = comments_df['cleaned_text'].tolist()
labels = comments_df['cluster'].tolist() # =========== whatever target column, what ever the hell that means?

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

tokenizer = RobertaTokenizer.from_pretrained("robert-base")

X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create datasets
train_dataset = TextDataset(X_train, y_train, tokenizer)
val_dataset = TextDataset(X_val, y_val, tokenizer)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model + training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaCNNClassifier(num_classes).to(device)

optimizer = Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Train
for epoch in range(5):
    loss = train_model(model, train_loader, optimizer, loss_fn, device)
    print(f"Epoch {epoch+1}: Training loss = {loss:.4f}")