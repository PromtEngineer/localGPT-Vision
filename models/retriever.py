# models/retriever.py

import base64
import os
from logger import get_logger

logger = get_logger(__name__)

def retrieve_documents(RAG, query, session_id, k=3):
    """
    Retrieves relevant documents based on the user query using Byaldi.

    Args:
        RAG (RAGMultiModalModel): The RAG model with the indexed documents.
        query (str): The user's query.
        session_id (str): The session ID to store images in per-session folder.
        k (int): The number of documents to retrieve.

    Returns:
        list: A list of image filenames corresponding to the retrieved documents.
    """
    try:
        logger.info(f"Retrieving documents for query: {query}")
        results = RAG.search(query, k=k)
        images = []
        session_images_folder = os.path.join('static', 'images', session_id)
        os.makedirs(session_images_folder, exist_ok=True)
        for result in results:
            if result.base64:
                image_data = base64.b64decode(result.base64)
                image_filename = f"retrieved_{result.doc_id}_{result.page_num}.png"
                image_path = os.path.join(session_images_folder, image_filename)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                images.append(os.path.join('images', session_id, image_filename))
                logger.debug(f"Retrieved and saved image: {image_filename}")
            else:
                # Handle cases where base64 data is not available
                logger.warning(f"No base64 data for document {result.doc_id}, page {result.page_num}")
        logger.info(f"Total {len(images)} documents retrieved.")
        return images
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []
