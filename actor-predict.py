import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PyPDF2 import PdfReader
import numpy as np

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    content = ''
    for page in reader.pages:
        content += page.extract_text() 
    return content

def prepare_dataframe_from_pdf(pdf_paths):
    data = []
    for pdf in pdf_paths:
        content = parse_pdf(pdf)
        data.append(content) 
    return pd.DataFrame(data, columns=['Falsehood Context'])

def get_all_pdfs_in_directory(directory):
    pdf_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    return pdf_paths

tokenizer = BertTokenizer.from_pretrained('tokenizer/bert/best_pofma_model_acc_0.826_f1_0.646_threshold_0.5')  
model = BertForSequenceClassification.from_pretrained('models/bert/best_pofma_model_acc_0.826_f1_0.646_threshold_0.5')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

actor_columns = ['Actor: Media', 
                 'Actor: Political Group or Figure', 
                 'Actor: Civil Society Group or Figure',
                 'Actor: Social Media Platform', 
                 'Actor: Internet Access Provider', 
                 'Actor: Private Individual'
                 ]

pdf_directory = 'data/pofma-media-notices'
pdf_paths = get_all_pdfs_in_directory(pdf_directory)

df = prepare_dataframe_from_pdf(pdf_paths)
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

pdf_dataset = POFMADataset(df['Falsehood Context'].values, tokenizer)
pdf_loader = torch.utils.data.DataLoader(pdf_dataset, batch_size=1, shuffle=False)

# Define a threshold for similarity (to handle similar scores)
similarity_threshold = 0.3

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

predictions_df = pd.DataFrame(predictions, columns=actor_columns)


df = df.reset_index(drop=True)  # indices match before concatenation
df[actor_columns] = predictions_df  # add predictions to respective columns
df.to_excel("inference/pofma_predictions.xlsx", index=False)

file_path = 'inference/pofma_predictions.xlsx'
df = pd.read_excel(file_path)

actors = ['Actor: Media', 'Actor: Political Group or Figure', 'Actor: Civil Society Group or Figure',
          'Actor: Social Media Platform', 'Actor: Internet Access Provider', 'Actor: Private Individual']

percentages = (df[actors].mean() * 100).round(2)
percentages_df = pd.DataFrame(percentages, columns=['Percentage of Notices'])

print(percentages_df)