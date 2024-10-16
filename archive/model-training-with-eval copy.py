import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. Load the Excel file and combine the first two columns
file_path = 'a1/cleaned master dataset.xlsx'  # Update this with your file path
df = pd.read_excel(file_path)

# Combine first two columns into "Falsehood Context"
df['Falsehood Context'] = df[df.columns[0]].astype(str) + " " + df[df.columns[1]].astype(str)

# Define actor columns and drop all other columns except actor columns and "Falsehood Context"
actor_columns = ['Actor: Media', 'Actor: Political Group or Figure', 'Actor: Civil Society Group or Figure',
                 'Actor: Social Media Platform', 'Actor: Internet Access Provider', 'Actor: Private Individual']

df = df[['Falsehood Context'] + actor_columns]

# Convert NaN values in actor columns to 0
df[actor_columns] = df[actor_columns].fillna(0)

# Filter out rows with no actors for training (sum of actor columns == 0)
df_actors_present = df[df[actor_columns].sum(axis=1) > 0].reset_index(drop=True)
df_no_actors = df[df[actor_columns].sum(axis=1) == 0].reset_index(drop=True)

# Prepare the labels as arrays (multi-label classification)
df_actors_present['Label'] = df_actors_present[actor_columns].values.tolist()

# 2. Split the data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_actors_present['Falsehood Context'], df_actors_present['Label'], test_size=0.2, random_state=42
)

# 3. Dataset Class Definition
class POFMADataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the training and validation datasets and loaders
train_dataset = POFMADataset(train_texts.tolist(), train_labels, tokenizer)
val_dataset = POFMADataset(val_texts.tolist(), val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 4. Model, Optimizer, and Loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(actor_columns))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.BCEWithLogitsLoss()

# 5. Training Loop with Accuracy Metric
num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Training Phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits).cpu().detach().numpy()

        # Convert probs to binary predictions (using threshold of 0.5)
        predictions = (probs > 0.5).astype(int)

        # Update accuracy count
        train_correct += (predictions == labels.cpu().numpy()).all(axis=1).sum()
        train_total += labels.size(0)
    
    train_accuracy = train_correct / train_total
    print(f"Training Loss: {train_loss/len(train_loader)}, Training Accuracy: {train_accuracy}")

    # Validation Phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            val_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()

            # Convert probs to binary predictions
            predictions = (probs > 0.5).astype(int)

            # Update accuracy count
            val_correct += (predictions == labels.cpu().numpy()).all(axis=1).sum()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    print(f"Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}")

# Save the model after training
model.save_pretrained('bert_pofma_model')
