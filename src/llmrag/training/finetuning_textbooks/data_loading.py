from PyPDF2 import PdfReader
from glob import glob
import re
import os
import json
from datasets import load_dataset

def PDFtext_to_chunks(pdf_path, skip_start_pages, skip_last_pages, header_lines, footer_lines):
    """
    Extract and preprocess text from a PDF.

    Args:
        pdf_path (str): Path to the PDF file.
        skip_start_pages (int): Number of pages to skip at the start.
        skip_last_pages (int): Number of pages to skip at the end.
        header_lines (int): Number of header lines to remove from each page.
        footer_lines (int): Number of footer lines to remove from each page.

    Returns:
        list: List of dictionaries with metadata and cleaned text chunks.
    """
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"Total Pages in Document: {num_pages}")
    all_chunks = []

    for page_num in range(skip_start_pages, num_pages - skip_last_pages):
        page = reader.pages[page_num]
        page_text = page.extract_text()

        # Split the page into lines and remove header/footer lines
        if header_lines == 0 and footer_lines == 0:
            lines = page_text.splitlines(True)
        elif header_lines == 0 and footer_lines != 0:
            lines = page_text.splitlines(True)[:-footer_lines]
        elif header_lines != 0 and footer_lines == 0:
            lines = page_text.splitlines(True)[header_lines:]
        else:
            lines = page_text.splitlines(True)[header_lines:-footer_lines]

        # Preprocess lines
        lines_modified = []
        for line in lines:
            line = line.strip()
            line = re.sub(r'[^\x00-\x7F]+', '', line)
            line = re.sub(r'[^\w\s.,!?\'"-]', '', line)
            line = re.sub(r'\s+', ' ', line)
            if line:
                lines_modified.append(line)
        
        # Combine lines into a single string
        cleaned_text = " ".join(lines_modified)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        # Split the cleaned text into chunks
        chunk_size = 512  # Specify the chunk size
        chunks = [cleaned_text[i:i + chunk_size] for i in range(0, len(cleaned_text), chunk_size)]

        # Create metadata and save chunks
        for chunk in chunks:
            if len(chunk) > 256:
                all_chunks.append({
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page_number": page_num + 1  # Page numbers start from 1
                    },
                    "content": chunk
                })

    return all_chunks

def get_pdf_config(file_path:str, index:int):
    with open(file_path, "r") as config_file:
        pdf_config = json.load(config_file)
    
    # Get configuration settings
    skip_start_pages = pdf_config[index].get(f"file{index+1}.pdf").get("skip_start_pages")
    skip_last_pages = pdf_config[index].get(f"file{index+1}.pdf").get("skip_last_pages")
    
    header_lines = pdf_config[index].get(f"file{index+1}.pdf").get("header_lines")
    footer_lines = pdf_config[index].get(f"file{index+1}.pdf").get("footer_lines")
    
    return skip_start_pages, skip_last_pages, header_lines, footer_lines

def chunks_to_jsonl(data, output_file):
    """
    Save a list of dictionaries to a JSONL file. Appends to the file if it exists,
    or creates a new file if it doesn't.

    Args:
        data (list): List of dictionaries to save.
        output_file (str): Path to the output JSONL file.
    """
    # Determine mode: 'a' for append if file exists, 'w' for write if it doesn't
    mode = 'a' if os.path.exists(output_file) else 'w'
    
    with open(output_file, mode, encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def loading_dataset(file_path):
    
    dataset = load_dataset("json",
                        data_files=file_path)
    
    print(f"Original Dataset: {dataset}\n")
    
    split_dataset = dataset["train"].train_test_split(test_size=0.10, seed=42)

    train_dataset = split_dataset["train"]
    print(f"Training Dataset: {train_dataset}\n")
    test_dataset = split_dataset["test"]
    print(f"Validation Dataset: {test_dataset}\n")

    return train_dataset, test_dataset

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["content"])

def group_texts(examples, block_size=256):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result