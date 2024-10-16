import os
import re
import torch
import pandas as pd
from transformers import pipeline
import PyPDF2

# Define the folder where the PDFs are stored
folder_path = 'POFMA Media Notices'
output_csv = 'classified_actors.csv'

# Load a pre-trained text classification model from Hugging Face
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define classification labels
classifications = ["Media", "Political Group or Figure", "Civil Society Group or Figure", 
                   "Social Media Platform", "Internet Access Provider", "Private Individual"]

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to find potential actors in a PDF
def find_actors_in_pdf(text):
    # Simple regex pattern to extract sentences containing actor references (adjust as needed)
    actor_patterns = re.findall(r"Facebook page,? (.+?) and|YouTube channel (.+?)\.", text, re.IGNORECASE)
    actors = [actor for pair in actor_patterns for actor in pair if actor]
    return actors

# Function to classify actors using the LLM
def classify_actors(actors):
    actor_classifications = {}
    for actor in actors:
        result = classifier(actor, classifications)
        best_classification = result['labels'][0]  # Get the top prediction
        actor_classifications[actor] = best_classification
    return actor_classifications

# Main function to process all PDFs in a folder and return data for CSV export
def process_pdfs_in_folder(folder_path):
    csv_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            text = extract_text_from_pdf(pdf_path)
            actors = find_actors_in_pdf(text)
            if actors:
                classified_actors = classify_actors(actors)
                for actor, classification in classified_actors.items():
                    csv_data.append({
                        "PDF File": filename,
                        "Actor": actor,
                        "Classification": classification
                    })
    return csv_data

# Save the extracted data to a CSV file
def export_to_csv(csv_data, output_csv):
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv, index=False)
    print(f"Data exported to {output_csv}")

# Example usage
parsed_actors = process_pdfs_in_folder(folder_path)
export_to_csv(parsed_actors, output_csv)
