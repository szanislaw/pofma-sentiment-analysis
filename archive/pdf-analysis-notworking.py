from transformers import pipeline
from huggingface_hub import login
import fitz  # PyMuPDF
import pandas as pd
import os

login(token="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")

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
    question = "Who is involved?"
    prompt = f"{text}\n\n{question}"
    pipe = pipeline("text-generation", device=0, model="Falconsai/text_summarization")
    generated_text = pipe(prompt, max_length=3000, num_return_sequences=1, batch_size=32)  # Reduced batch size
    return generated_text[0]['generated_text']

# Main function to process all PDF files
def main():
    folder_name = "POFMA Media Notices"
    ensure_directory_exists(folder_name)
    
    pdf_files = [f for f in os.listdir(folder_name) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the folder.")
        return
    
    all_entities = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_name, pdf_file)
        print(f"Processing file: {pdf_file}")
        
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"No text extracted from: {pdf_file}")
            continue
        
        print(f"Extracted text from {pdf_file}: {text[:500]}...")  # Print first 500 characters for brevity
        
        try:
            generated_text = generate_text_with_mistral(text)
            print(f"Generated text for {pdf_file}: {generated_text.strip()}")
        except Exception as e:
            print(f"Error generating text for {pdf_file}: {e}")
            continue
        
        # Collect results
        all_entities.append({
            "file": pdf_file,
            "generated_text": generated_text.strip(),
        })
    
    if not all_entities:
        print("No entities extracted from any PDF.")
        return
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_entities)
    output_folder = "output"
    ensure_directory_exists(output_folder)
    output_path = os.path.join(output_folder, 'POFMA_Entity_Extraction.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
