import os
import datetime
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 1. Load the Excel file and combine the first two columns
file_path = 'model/cleaned master dataset.xlsx'
df = pd.read_excel(file_path)

# Combine first two columns into "Falsehood Context"
df['Falsehood Context'] = df[df.columns[0]].astype(str) + " " + df[df.columns[1]].astype(str)

# Define actor columns and drop all other columns except actor columns and "Falsehood Context"
actor_columns = ['Actor: Media', 'Actor: Political Group or Figure', 'Actor: Civil Society Group or Figure', 'Actor: Social Media Platform', 'Actor: Internet Access Provider', 'Actor: Private Individual']
df = df[['Falsehood Context'] + actor_columns]

# Convert NaN values in actor columns to 0
df[actor_columns] = df[actor_columns].fillna(0)

# Filter out rows with no actors for training
df_actors_present = df[df[actor_columns].sum(axis=1) > 0].reset_index(drop=True)
df_no_actors = df[df[actor_columns].sum(axis=1) == 0].reset_index(drop=True)

# Prepare the labels as arrays (multilabel classification)
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

# 4. Tokenizer and model initialization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

# 5. Prepare Datasets and DataLoader
train_dataset = POFMADataset(train_texts.values, train_labels.values, tokenizer)
val_dataset = POFMADataset(val_texts.values, val_labels.values, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 6. Define the training function with Gradient Clipping
def train_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

    return total_loss / len(dataloader)

# 7. Define the evaluation function with accuracy calculation
def eval_model(model, dataloader, device, threshold=0.5):
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

            # Get logits and convert to probabilities using sigmoid
            logits = outputs.logits
            probs = torch.sigmoid(logits)

            # Convert probabilities to binary predictions using the threshold
            preds = (probs > threshold).float()

            # Count correct predictions
            correct_predictions += (preds == labels).sum().item()  # Count correct labels
            total_predictions += torch.numel(labels)  # Total number of labels
            
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0)

    accuracy = correct_predictions / total_predictions
    return total_loss / len(dataloader), accuracy, f1

# 8. Set up device, optimizer, and start training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimizer Settings
# optimizer = Adafactor(model.parameters(), lr=1e-4, scale_parameter=False, relative_step=False)
# optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.005)  # Experiment with 3e-5 as well

# 9. Training Loop with Early Stopping and Gradient Clipping
best_f1 = 0
patience = 10
patience_counter = 0

epochs = 30
f1_scores = []
val_accuracies = []

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    
    # Gradient CLipping
    train_loss = train_epoch(model, train_loader, optimizer, device, max_grad_norm=1.0)
    val_loss, val_accuracy, f1 = eval_model(model, val_loader, device)
    f1_scores.append(f1)
    val_accuracies.append(val_accuracy)
    
    print(f'Train Loss: {train_loss:.3f}')
    print(f'Val Loss: {val_loss:.3f}')
    print(f'Val Accuracy: {val_accuracy:.3f}')
    print(f'F1 Score: {f1:.3f}')
    
    # Early Stopping
    if f1 > best_f1:
        best_f1 = f1
        best_val_accuracy = val_accuracy 
        patience_counter = 0
        
        # Save the model
        model_file_name = f"models/best_pofma_model_acc_{best_val_accuracy:.3f}_f1_{best_f1:.3f}"
        model.save_pretrained(model_file_name)
        tokenizer.save_pretrained(model_file_name)
        
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# # Define the graph folder and make it
# output_dir = 'model-graph'
# os.makedirs(output_dir, exist_ok=True)

# # Plotting the F1 scores
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, epochs + 1), f1_scores, marker='o', linestyle='-', color='b')
# plt.title('Epoch vs F1 Score')
# plt.xlabel('Epoch')
# plt.ylabel('F1 Score')
# plt.grid(True)

# current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# file_name = f'epoch_vs_f1_score_{current_time}.png'
# file_path = os.path.join(output_dir, file_name)  # Create the full file path
# plt.savefig(file_path, format='png')
# plt.show() 

completed_epochs = len(f1_scores)
plt.figure(figsize=(8, 6))
plt.plot(range(1, completed_epochs + 1), f1_scores, marker='o', linestyle='-', color='b', label='F1 Score')
plt.plot(range(1, completed_epochs + 1), val_accuracies, marker='s', linestyle='-', color='r', label='Validation Accuracy')

# Add titles and labels
plt.title('Epoch vs F1 Score & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.grid(True)

# Add a legend to differentiate between F1 and validation accuracy
plt.legend()

# Save the plot to a file (if desired, including date and time)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = 'model-graph'
os.makedirs(output_dir, exist_ok=True)
file_name = f'epoch_vs_f1_and_val_acc_{current_time}.png'
file_path = os.path.join(output_dir, file_name)
plt.savefig(file_path, format='png')

# Display the plot
plt.show()


# Optional: Predict on rows without actors (eyeballing set)
eyeball_texts = df_no_actors['Falsehood Context'].values
eyeball_labels = df_no_actors[actor_columns].values.tolist()

eyeball_dataset = POFMADataset(eyeball_texts, eyeball_labels, tokenizer)
eyeball_loader = DataLoader(eyeball_dataset, batch_size=16, shuffle=False)

# Define a threshold for similarity
similarity_threshold = 0.2  # Adjust this threshold based on your needs

# Define actor columns
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

# Convert the predictions list to a DataFrame with columns corresponding to the actor columns
predictions_df = pd.DataFrame(predictions, columns=actor_columns)

# Add the predicted values back to the original dataframe (df_no_actors) in their respective columns
df_no_actors[actor_columns] = predictions_df

# Save the modified dataframe with predictions for review
df_no_actors.to_excel("inference/eyeball_predictions.xlsx", index=False)