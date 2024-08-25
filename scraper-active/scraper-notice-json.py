import os
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
import torch
import fitz  # PyMuPDF
import json

# URL of the POFMA Media Center Page
base_url = "https://www.pofmaoffice.gov.sg/media-centre/"

# Initialize the NER pipeline
device = 0 if torch.cuda.is_available() else -1
ner_pipeline = pipeline("ner", grouped_entities=True, device=device)

def get_page_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def clean_headline(headline):
    pattern = r'(\\b\\d{1,2}\\s?[A-Za-z]{3,9}\\s?\\d{2,4}\\b)|(\\b\\d{1,2}[A-Za-z]{3}\\d{2,4}\\b)|(\\b\\d{1,2}\\s?[A-Za-z]{3}\\s?\\d{2,4}\\b)'
    cleaned_headline = re.sub(pattern, '', headline).strip()
    cleaned_headline = re.sub(r'\\.{3,}', '', cleaned_headline).strip()
    return cleaned_headline

def extract_source(headline):
    entities = ner_pipeline(headline)
    if entities:
        if headline.split()[0] == "Minister":
            return entities[0]['word']
    return 'No Source'

def parse_headlines_links_dates(page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    headlines_links_dates = []
    
    for card in soup.find_all('a', class_='is-media-card'):
        headline_tag = card.find('h5')
        headline = headline_tag.get_text(strip=True) if headline_tag else 'No Headline'
        cleaned_headline = clean_headline(headline)
        source = extract_source(headline)
        
        partial_link = card['href']
        full_link = f"https://www.pofmaoffice.gov.sg{partial_link}"
        
        date = 'No Date'
        for small_tag in card.find_all('small', class_='has-text-white'):
            small_text = small_tag.get_text(strip=True)
            if small_text[0].isdigit():
                date = small_text
                break
        
        headlines_links_dates.append({
            'headline': cleaned_headline,
            'link': full_link,
            'date': date,
            'source': source
        })
    return headlines_links_dates

def download_pdfs(headlines_links_dates):
    folder_name = "POFMA Media Notices"
    os.makedirs(folder_name, exist_ok=True)
    
    all_files_exist = True
    for item in headlines_links_dates:
        pdf_name = os.path.join(folder_name, f"{item['headline']}.pdf")
        if not os.path.exists(pdf_name):
            all_files_exist = False
            break
    
    if all_files_exist:
        print("All PDF files already exist. Skipping download.")
        return
    
    for item in headlines_links_dates:
        pdf_name = os.path.join(folder_name, f"{item['headline']}.pdf")
        if not os.path.exists(pdf_name):
            pdf_url = item['link']
            response = requests.get(pdf_url)
            if response.status_code == 200:
                with open(pdf_name, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                print(f"Downloaded: {pdf_name}")
            else:
                print(f"Failed to download: {pdf_url}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def main():
    page_content = get_page_content(base_url)
    
    if page_content:
        headlines_links_dates = parse_headlines_links_dates(page_content)
        df = pd.DataFrame(headlines_links_dates)
        df.to_csv('POFMA Reduced Dataset.csv', index=False)
        print("The POFMA Media Releases have been successfully scraped and saved to POFMA Reduced Dataset.csv.")
        download_pdfs(headlines_links_dates)
        
        # Extract text from all PDFs and save to JSON
        folder_name = "POFMA Media Notices"
        pdf_files = [f for f in os.listdir(folder_name) if f.endswith('.pdf')]
        
        pdf_texts = {}
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_name, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            pdf_texts[pdf_file] = text
        
        with open('POFMA_PDF_Texts.json', 'w') as json_file:
            json.dump(pdf_texts, json_file, indent=4)
        
        print("Extracted text from all PDFs and saved to POFMA_PDF_Texts.json.")
    else:
        print("Error 01: Failed to retrieve the page content.")

if __name__ == "__main__":
    main()
