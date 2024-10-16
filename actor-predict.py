import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PyPDF2 import PdfReader
import numpy as np

# Function to parse PDF content
def parse_pdf(file_path):
    reader = PdfReader(file_path)
    content = ''
    for page in reader.pages:
        content += page.extract_text()  # Extract text from each page
    return content

# Function to prepare DataFrame for predictions
def prepare_dataframe_from_pdf(pdf_paths):
    data = []
    for pdf in pdf_paths:
        content = parse_pdf(pdf)
        data.append(content)  # Add content to the DataFrame
    return pd.DataFrame(data, columns=['Falsehood Context'])

# Function to get all PDF files in a directory
def get_all_pdfs_in_directory(directory):
    pdf_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    return pdf_paths

# Load the saved tokenizer and model
tokenizer = BertTokenizer.from_pretrained('pofma_tokenizer')  # Use your saved tokenizer directory
model = BertForSequenceClassification.from_pretrained('pofma_model')  # Use your saved model directory

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# List of actor columns (B-G)
actor_columns = ['Actor: Media', 'Actor: Political Group or Figure', 'Actor: Civil Society Group or Figure',
                 'Actor: Social Media Platform', 'Actor: Internet Access Provider', 'Actor: Private Individual']

# Directory containing the PDF files
pdf_directory = 'POFMA Media Notices'  # Change this to your actual PDF directory

# Get all PDFs in the directory
pdf_paths = get_all_pdfs_in_directory(pdf_directory)

# Prepare DataFrame from PDF files
df = prepare_dataframe_from_pdf(pdf_paths)

# Tokenize the PDF content for the model
class POFMADataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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
        }

# Create dataset and dataloader
pdf_dataset = POFMADataset(df['Falsehood Context'].values, tokenizer)
pdf_loader = torch.utils.data.DataLoader(pdf_dataset, batch_size=1, shuffle=False)

# Define a threshold for similarity (to handle similar scores)
similarity_threshold = 0.3

# Perform predictions
model.eval()
predictions = []
with torch.no_grad():
    for batch in pdf_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()  # Apply sigmoid to get probabilities
        
        for prob in probs:
            binary_labels = np.zeros_like(prob)
            max_idx = np.argmax(prob)
            binary_labels[max_idx] = 1
            
            for i, p in enumerate(prob):
                if i != max_idx and np.abs(p - prob[max_idx]) <= similarity_threshold:
                    binary_labels[i] = 1
            
            predictions.append(binary_labels)

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=actor_columns)

# Add the predictions to the original DataFrame without modifying 'Falsehood Context'
df = df.reset_index(drop=True)  # Ensure the indices match before concatenation
df[actor_columns] = predictions_df  # Add predictions to respective columns

# Save the results to an Excel file
df.to_excel("pdf_predictions.xlsx", index=False)
