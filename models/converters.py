# models/converters.py

import os
from docx2pdf import convert
from logger import get_logger

logger = get_logger(__name__)

def convert_docs_to_pdfs(folder_path):
    """
    Converts .doc and .docx files in the folder to PDFs.

    Args:
        folder_path (str): The path to the folder containing documents.
    """
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.doc', '.docx')):
                doc_path = os.path.join(folder_path, filename)
                pdf_path = os.path.splitext(doc_path)[0] + '.pdf'
                convert(doc_path, pdf_path)
                logger.info(f"Converted '{filename}' to PDF.")
    except Exception as e:
        logger.error(f"Error converting documents to PDFs: {e}")
        raise