import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load BERTweet model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=3)

def chunk_text(text, max_length=512):
    # Tokenize the text into tokens
    tokens = tokenizer.tokenize(text)
    
    # Split tokens into chunks
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    
    return [" ".join(chunk) for chunk in chunks]

def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return None  # Handle empty/invalid text
    
    # Chunk the text
    text_chunks = chunk_text(text, max_length=512)
    
    sentiments = []
    
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Perform sentiment analysis
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get the sentiment for this chunk
        sentiment = torch.argmax(probabilities, dim=-1).item()
        sentiments.append(sentiment)
    
    # Aggregate the sentiments from all chunks
    # Here, I'm using the majority sentiment; you could also calculate the average
    final_sentiment = max(set(sentiments), key=sentiments.count)
    
    return final_sentiment


# Base directories
input_base_dir = 'scraped-comments'
output_base_dir = 'scraped-comments-sentiment'

# Ensure the base output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Loop through each subfolder in the input directory
for subfolder in os.listdir(input_base_dir):
    input_folder = os.path.join(input_base_dir, subfolder)
    output_folder = os.path.join(output_base_dir, subfolder)
    
    # Check if the input_folder is a directory
    if os.path.isdir(input_folder):
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Loop through all CSV files in the current subfolder
        for csv_file in os.listdir(input_folder):
            if csv_file.endswith('.csv'):
                input_file_path = os.path.join(input_folder, csv_file)
                output_file_path = os.path.join(output_folder, csv_file)
                
                # Read the CSV file
                df = pd.read_csv(input_file_path)
                
                # Perform sentiment analysis
                df['sentiment'] = df['Comment'].apply(analyze_sentiment)  # Comment is for fb_comments.csv
                
                # Save the results to the corresponding output folder
                df.to_csv(output_file_path, index=False)
