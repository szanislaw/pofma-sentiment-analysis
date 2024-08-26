import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import fitz  # PyMuPDF
import os
import pandas as pd

# Define the functions
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def predict_NuExtract(model, tokenizer, text, schema, example=["","",""]):
    schema = json.dumps(json.loads(schema), indent=4)
    input_llm =  "<|input|>\n### Template:\n" +  schema + "\n"
    for i in example:
        if i != "":
            input_llm += "### Example:\n"+ json.dumps(json.loads(i), indent=4)+"\n"
    
    input_llm +=  "### Text:\n"+text +"\n<|output|>\n"
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000).to("cuda")

    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    return output.split("<|output|>")[1].split("<|end-output|>")[0]

def process_folder(folder_path, model, tokenizer, schema):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            prediction = predict_NuExtract(model, tokenizer, text, schema)
            results.append((filename, prediction))
    return results

def parse_prediction(prediction):
    """Parses the JSON prediction string to extract required fields."""
    try:
        prediction_json = json.loads(prediction)
        name = prediction_json.get("Offending Actor", {}).get("Name", "")
        organisation = prediction_json.get("Offending Actor", {}).get("Organisation", "")
        offense_date = prediction_json.get("Offense Date", "")
        return name, organisation, offense_date
    except json.JSONDecodeError:
        return "", "", ""

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)

model.to("cuda")
model.eval()

# Path to the folder containing PDF files
folder_path = 'POFMA Media Notices/arch'  # Replace with your actual folder path

# Define the schema
schema = """{
    "Offending Actor": {
        "Name": "",
        "Organisation": ""
    },
    "Offense Date": "",
    "Nature of Offense": ""
}"""

# Process each PDF in the folder
results = process_folder(folder_path, model, tokenizer, schema)

# Extract data and create a DataFrame
data = []
for filename, prediction in results:
    name, organisation, offense_date = parse_prediction(prediction)
    data.append({
        "PDF Name": filename,
        "Name": name,
        "Organisation": organisation,
        "Offense Date": offense_date
    })

df = pd.DataFrame(data)
df.to_csv('extracted_data.csv', index=False)
print(df.head())
