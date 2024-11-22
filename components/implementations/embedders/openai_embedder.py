from components.interfaces.all_interfaces import Embedder

import time
import random
import openai
import logging
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class OpenAIEmbedder(Embedder):
    def __init__(self, 
                 openai_api_key: str, 
                 embedding_model: str, 
                 batch_size: int, 
                 rate_limit_config: dict):
        self.batch_size = batch_size
        self.rate_limit_config = rate_limit_config
        self.logger = logging.getLogger('Embedder')
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=embedding_model
        )

    def embed(self, documents: List["Document"]) -> List[float]:
        """Embeds documents with retry mechanism for rate limiting."""
        texts = [doc.content for doc in documents]
        max_retries = self.rate_limit_config.get("max_retries")
        initial_delay = self.rate_limit_config.get("initial_delay")
        max_delay = self.rate_limit_config.get("max_delay")

        attempt = 0
        while attempt < max_retries:
            try:
                return self.embeddings.embed_documents(texts)
            except openai.error.RateLimitError as e:
                attempt += 1
                delay = min(initial_delay * (2 ** attempt) * (1 + random.uniform(0, 0.1)), max_delay)
                self.logger.warning(f"Rate limit exceeded on attempt {attempt}. Retrying in {delay:.2f} seconds... ({e})")
                time.sleep(delay)
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt}: {e}")
                raise  # Raise other exceptions immediately

        raise Exception("Exceeded maximum retries for embedding documents due to rate limit.")
