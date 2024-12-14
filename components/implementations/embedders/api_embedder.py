from components.interfaces.all_interfaces import Embedder

import logging
import requests
from typing import List
from langchain_core.documents import Document


class GeneralEmbeddingAPIClient(Embedder):
    def __init__(
        self,
        host: str, 
        port: int
    ):

        self.host = host
        self.port = port

        # Configure logger
        self.logger = logging.getLogger('Embedder')

    def embed(self, documents: List["Document"]) -> List[List[float]]:
        self.logger.info("Embedding starts")
        all_vectors = []

        for doc in documents:
            vec = self.embed_query(doc.page_content)
            if vec is not None:
                all_vectors.append(vec)

        return all_vectors


    def embed_query(self, query: str) -> List[float]:
        if not query:
            self.logger.error("No query provided for embedding.")
            return None

        url = f"http://{self.host}:{self.port}/embed_query"
        headers = {'Content-Type': 'application/json'}
        payload = {"text": query}

        try:
            
            response = requests.post(url, json=payload, headers=headers)
        
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Error: Received status code {response.status_code}. Response: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"An error occurred while making the request: {e}")
            return None
