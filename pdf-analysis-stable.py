import fitz  # PyMuPDF
from transformers import pipeline
from huggingface_hub import login

login(token="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")

# Step 1: Extract Text from PDF using fitz (PyMuPDF)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Step 2: Load HuggingFace Summarization Model
summarizer = pipeline("summarization", model="microsoft/Phi-3-mini-128k-instruct", tokenizer="microsoft/Phi-3-mini-128k-instruct")

def summarize_text(text, prompt, max_length=130, min_length=30):
    prompt_text = f"{prompt}\n\n{text}"
    summaries = summarizer(prompt_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summaries[0]['summary_text']

# Path to your PDF document
pdf_path = 'POFMA Media Notices\Alternate Authority for the Minister for Education Instructs POFMA Office to Issue Correction Directions.pdf'

# Define the prompt
prompt = "Please provide a concise summary of the following document:"

# Extract text from PDF
text = extract_text_from_pdf(pdf_path)

# Summarize the extracted text with the prompt
summary = summarize_text(text, prompt)

# Print the summary
print(summary)
