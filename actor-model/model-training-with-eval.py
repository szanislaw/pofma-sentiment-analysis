import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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
        labels = torch.tensor(self.labels[idx], dtype=torch.float)  # Ensure labels are float for multi-label
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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# 5. Prepare Datasets and DataLoader
train_dataset = POFMADataset(train_texts.values, train_labels.values, tokenizer)
val_dataset = POFMADataset(val_texts.values, val_labels.values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 6. Define the training function
def train_epoch(model, dataloader, optimizer, device):
    model = model.train()
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
        optimizer.step()

    return total_loss / len(dataloader)

# 7. Define the evaluation function
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

    accuracy = correct_predictions / total_predictions
    return total_loss / len(dataloader), accuracy


# 8. Set up device, optimizer, and start training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
from transformers import Adafactor
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 9. Training Loop
epochs = 30
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_accuracy = eval_model(model, val_loader, device)
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train Loss: {train_loss:.3f}')
    print(f'Val Loss: {val_loss:.3f}')
    print(f'Val Accuracy: {val_accuracy:.3f}')


# 10. Save the model after training
model.save_pretrained('pofma_model')
tokenizer.save_pretrained('pofma_tokenizer')

# Optional: Predict on rows without actors (eyeballing set)
eyeball_texts = df_no_actors['Falsehood Context'].values
eyeball_labels = df_no_actors[actor_columns].values.tolist()

eyeball_dataset = POFMADataset(eyeball_texts, eyeball_labels, tokenizer)
eyeball_loader = DataLoader(eyeball_dataset, batch_size=8, shuffle=False)

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
df_no_actors.to_excel("eyeball_predictions.xlsx", index=False)