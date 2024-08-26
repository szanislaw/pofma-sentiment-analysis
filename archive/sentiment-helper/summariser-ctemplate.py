import fitz  # PyMuPDF
from transformers import pipeline
from fpdf import FPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def summarize_text(text, max_length=150, min_length=50):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def save_summary_to_pdf(summary, output_pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf.output(output_pdf_path)

def main(pdf_path, output_pdf_path):
    text = extract_text_from_pdf(pdf_path)
    summary = summarize_text(text)
    save_summary_to_pdf(summary, output_pdf_path)
    print(f"Summary saved to {output_pdf_path}")

if __name__ == "__main__":
    input_pdf_path = "input.pdf"  # Replace with your input PDF file path
    output_pdf_path = "summary.pdf"  # Replace with your desired output PDF file path
    main(input_pdf_path, output_pdf_path)
