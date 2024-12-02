import os
import datetime
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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
        labels = torch.tensor(self.labels[idx], dtype=torch.float) 
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

def train_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

    return total_loss / len(dataloader)

def eval_model(model, dataloader, device, threshold):
    model = model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
            correct_predictions += (preds == labels).sum().item()  # correct labels
            total_predictions += torch.numel(labels)  # total number of labels
            
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0)

    accuracy = correct_predictions / total_predictions
    return total_loss / len(dataloader), accuracy, f1, threshold

#--------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = 'model/cleaned master dataset.xlsx'

actor_columns = ['Actor: Media', 
                 'Actor: Political Group or Figure', 
                 'Actor: Civil Society Group or Figure', 
                 'Actor: Social Media Platform', 
                 'Actor: Internet Access Provider', 
                 'Actor: Private Individual'
                 ]

df = pd.read_excel(file_path)
df['Falsehood Context'] = df[df.columns[0]].astype(str) + " " + df[df.columns[1]].astype(str)
df = df[['Falsehood Context'] + actor_columns]
df[actor_columns] = df[actor_columns].fillna(0)
df_actors_present = df[df[actor_columns].sum(axis=1) > 0].reset_index(drop=True)
df_no_actors = df[df[actor_columns].sum(axis=1) == 0].reset_index(drop=True)
df_actors_present['Label'] = df_actors_present[actor_columns].values.tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_actors_present['Falsehood Context'], df_actors_present['Label'], test_size=0.2, random_state=42
)
        
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

train_dataset = POFMADataset(train_texts.values, train_labels.values, tokenizer)
val_dataset = POFMADataset(val_texts.values, val_labels.values, tokenizer)

# adjust batch size
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

class_weights = {}
for actor in actor_columns:
    class_count = df_actors_present[actor].value_counts(normalize=True)
    weight = class_count[0] / class_count[1] if 1 in class_count else 1.0
    class_weights[actor] = weight

class_weights_tensor = torch.tensor(list(class_weights.values())).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)

model = model.to(device)

# Optimizer Settings
# optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
# optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1) 
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
# optimizer = Adafactor(model.parameters(), lr=1e-4, scale_parameter=False, relative_step=False)
# optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)

best_f1 = 0
best_val_accuracy = 0
best_val_loss = float('inf')
patience_cnt = 0    

patience = 15
threshold = 0.5
epochs = 50

f1_scores = []
val_accuracies = []
val_losses = []

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')

    train_loss = train_epoch(model, train_loader, optimizer, device, max_grad_norm=1.0)
    val_loss, val_accuracy, f1, threshold = eval_model(model, val_loader, device, threshold)
    f1_scores.append(f1)
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)
    
    print(f'Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}, F1 Score: {f1:.3f}')
    
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_f1 = f1
        best_val_accuracy = val_accuracy 
        patience_cnt = 0
        
    else:
        patience_cnt += 1
    
    if patience_cnt >= patience:
        print("Early stopping triggered.")
        break
        
    model_file_name = f"models/bert/best_pofma_model_acc_{best_val_accuracy:.3f}_f1_{best_f1:.3f}_threshold_{threshold}"
    tokenizer_file_name = f"tokenizer/bert/best_pofma_model_acc_{best_val_accuracy:.3f}_f1_{best_f1:.3f}_threshold_{threshold}"
    model.save_pretrained(model_file_name)
    tokenizer.save_pretrained(tokenizer_file_name)
    

completed_epochs = len(f1_scores)

#figure
plt.figure(figsize=(8, 6))
plt.plot(range(1, completed_epochs + 1), f1_scores, marker='o', linestyle='-', color='b', label='F1 Score')
plt.plot(range(1, completed_epochs + 1), val_accuracies, marker='s', linestyle='-', color='r', label='Validation Accuracy')
plt.plot(range(1, completed_epochs + 1), val_losses, marker='x', linestyle='-', color='g', label='Validation Loss')
plt.title(f'Epoch vs F1 Score & Validation Accuracy, Threshold: {threshold}')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.grid(True)
plt.legend()

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = 'model-graph'
os.makedirs(output_dir, exist_ok=True)
file_name = f'epoch_vs_f1_and_val_acc_model_acc_{best_val_accuracy:.3f}_f1_{best_f1:.3f}_threshold_{threshold}.png'
file_path = os.path.join(output_dir, file_name)
plt.savefig(file_path, format='png')

plt.show()

# EYEBALL SET ----------------------------------------------------------------------------------------------------- 
eyeball_texts = df_no_actors['Falsehood Context'].values
eyeball_labels = df_no_actors[actor_columns].values.tolist()

eyeball_dataset = POFMADataset(eyeball_texts, eyeball_labels, tokenizer)
eyeball_loader = DataLoader(eyeball_dataset, batch_size=16, shuffle=False)

similarity_threshold = 0.3  # Adjust this threshold based on your needs

actor_columns = ['Actor: Media', 'Actor: Political Group or Figure', 'Actor: Civil Society Group or Figure',
                 'Actor: Social Media Platform', 'Actor: Internet Access Provider', 'Actor: Private Individual']

# Perform predictions
model.eval()
predictions = []
with torch.no_grad():
    for batch in eyeball_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()  # Apply sigmoid to get probabilities
        
        for prob in probs:
            # Set all values to 0 by default
            binary_labels = np.zeros_like(prob)
            
            # Find the index of the maximum score
            max_idx = np.argmax(prob)
            
            # Set the max score to 1
            binary_labels[max_idx] = 1
            
            # Check for similar scores within the threshold
            for i, p in enumerate(prob):
                if i != max_idx and np.abs(p - prob[max_idx]) <= similarity_threshold:
                    binary_labels[i] = 1
            
            predictions.append(binary_labels)

predictions_df = pd.DataFrame(predictions, columns=actor_columns)

df_no_actors[actor_columns] = predictions_df
df_no_actors.to_excel("inference/eyeball_predictions.xlsx", index=False)