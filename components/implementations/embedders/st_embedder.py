import os
import logging
from typing import List
import torch

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from components.interfaces.all_interfaces import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self,
                 embedding_model):
        """
        Sentence Transformer Embedder
        
        **Sentence Transformer Models:**
        - [Sentence Transformers](https://sbert.net/)
        - [MTEB Leaderboards](https://huggingface.co/spaces/mteb/leaderboard)

        """
        self.embedding_model = embedding_model
        self.logger = logging.getLogger('Embedder')


        # load model
        try:
            self.model = SentenceTransformer(self.embedding_model, trust_remote_code=True, device="cuda")
            self.logger.info(f"{self.embedding_model} loaded successfully.")
            self.logger.info("Max Sequence Length:", self.model.max_seq_length, "changing it to 512")
            self.model.max_seq_length = 512

        except Exception as e:
            self.logger.error(f"Failed to load {self.embedding_model} model: {e}")
            raise


    def embed(self, documents: List[Document]) -> List[List[float]]:
        if not documents:
            self.logger.error("No documents provided for embedding.")
            os._exit(199)

        batch_size = 6000
        embedding_vectors = list()

        batches = (len(documents) // batch_size) + 1
            
        for i in range(batches):
            range_start = i*batch_size
            range_end = i*batch_size+batch_size

            dox = [doc.page_content for doc in documents[range_start:range_end]]
            
            embeddings_nparray = self.model.encode(dox, batch_size=16)
            
            embedding_vectors.extend(embeddings_nparray.tolist())
            del embeddings_nparray
            del dox
            torch.cuda.empty_cache()
        
        self.logger.info(f"docs embedded successfully, embeddings type: {type(embedding_vectors)}")
        return embedding_vectors



