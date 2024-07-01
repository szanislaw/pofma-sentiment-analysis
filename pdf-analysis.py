# Use a pipeline as a high-level helper
from transformers import pipeline
from huggingface_hub import login
import fitz  # PyMuPDF
import pandas as pd

login(token="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")

import os
import fitz  # PyMuPDF
import pandas as pd
from transformers import pipeline

# Ensure the necessary directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to generate text using Mistral
def generate_text_with_mistral(text):
    question = "Who is involved? You only need to answer this question with the entity involved and nothing else."
    prompt = f"{text}\n\n{question}"
    pipe = pipeline("text-generation", device=0, model="mistralai/Mistral-7B-Instruct-v0.3")
    generated_text = pipe(prompt, max_length=3000, num_return_sequences=1)
    return generated_text[0]['generated_text']

# Main function to process all PDF files
def main():
    folder_name = "POFMA Media Notices"
    ensure_directory_exists(folder_name)
    
    pdf_files = [f for f in os.listdir(folder_name) if f.endswith('.pdf')]
    
    all_entities = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_name, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        generated_text = generate_text_with_mistral(text)
        
        # Collect results
        all_entities.append({
            "file": pdf_file,
            "generated_text": generated_text.strip(),
        })
        
        print(generated_text.strip())
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_entities)
    output_folder = "output"
    ensure_directory_exists(output_folder)
    output_path = os.path.join(output_folder, 'POFMA_Entity_Extraction.csv')
    df.to_csv(output_path, index=False)
    print(f"Entity extraction completed and saved to {output_path}.")

if __name__ == "__main__":
    main()
