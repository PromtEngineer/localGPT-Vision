# models/retriever.py

import base64
import os
from PIL import Image
from io import BytesIO
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
        for i, result in enumerate(results):
            if result.base64:
                image_data = base64.b64decode(result.base64)
                image = Image.open(BytesIO(image_data))
                image_filename = f"retrieved_{i}.png"
                image_path = os.path.join(session_images_folder, image_filename)
                image.save(image_path, format='PNG')
                images.append(os.path.join('images', session_id, image_filename))
                logger.debug(f"Retrieved and saved image: {image_filename}")
            else:
                logger.warning(f"No base64 data for document {result.doc_id}, page {result.page_num}")
        logger.info(f"Total {len(images)} documents retrieved.")
        return images
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []