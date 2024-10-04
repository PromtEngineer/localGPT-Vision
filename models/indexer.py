# models/indexer.py

import os
from byaldi import RAGMultiModalModel
from models.converters import convert_docs_to_pdfs
from logger import get_logger

logger = get_logger(__name__)

def index_documents(folder_path, index_name='document_index', index_path=None, indexer_model='vidore/colpali'):
    """
    Indexes documents in the specified folder using Byaldi.

    Args:
        folder_path (str): The path to the folder containing documents to index.
        index_name (str): The name of the index to create or update.
        index_path (str): The path where the index should be saved.
        indexer_model (str): The name of the indexer model to use.

    Returns:
        RAGMultiModalModel: The RAG model with the indexed documents.
    """
    try:
        logger.info(f"Starting document indexing in folder: {folder_path}")
        # Convert non-PDF documents to PDFs
        convert_docs_to_pdfs(folder_path)
        logger.info("Conversion of non-PDF documents to PDFs completed.")

        # Initialize RAG model
        RAG = RAGMultiModalModel.from_pretrained(indexer_model)
        if RAG is None:
            raise ValueError(f"Failed to initialize RAGMultiModalModel with model {indexer_model}")
        logger.info(f"RAG model initialized with {indexer_model}.")

        # Index the documents in the folder
        RAG.index(
            input_path=folder_path,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )

        logger.info(f"Indexing completed. Index saved at '{index_path}'.")

        return RAG
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise