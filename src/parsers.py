# src/parsers.py
import os
import json
import logging
import pandas as pd
import pdfplumber

logger = logging.getLogger("FoodMetriX_Core.Parsers")

def parse_to_dataframe(uploaded_file):
    """Extracts data from tabular structures (Excel/CSV) into a standard Pandas DataFrame."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(uploaded_file)
    elif file_extension == '.csv':
        return pd.read_csv(uploaded_file)
    else:
        raise TypeError(f"Unsupported tabular format: {file_extension}")

def parse_pdf_text(uploaded_file):
    """Extracts raw text strings out of complex multi-page PDF documents."""
    chunks = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                for chunk_idx, block in enumerate(text.split("\n\n")):
                    if block.strip():
                        chunks.append({
                            "text": block.strip(),
                            "meta": f"Page {page_num+1}, Chunk {chunk_idx+1}"
                        })
    return chunks