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
                 model: str, 
                 batch_size: int):
        
        self.model = model
        self.batch_size = batch_size
        
        self.logger = logging.getLogger('Embedder')

    def embed(self, documents: List["Document"]) -> List[List[float]]:

        texts = []
        
        self.logger.info("Embedding starts")
        try:
            for doc in documents:
                texts.append(doc.page_content)
        except Exception as e:
            self.logger.error(f"Cannot get content from documents: {e}")

        # get the values from dict (to be implemented later - we could not read out the dict from config)
        max_retries = 2
        initial_delay = 1.0
        max_delay = 60.0

        model  = str(self.model)

        embeddings = OpenAIEmbeddings(model=model)


        vectors = embeddings.embed_documents(texts)
        self.logger.info(f"Documents got embedded to {len(vectors)} vectors")
        return vectors

        # could not find openai errors & 404
        # attempt = 0
        # while attempt < max_retries:
        #     try:
        #         vectors = embeddings.embed_documents(texts)
        #         self.log.info(f"Documents got embedded to {len(vectors)} vectors")
        #         return vectors
        #     except openai.error.RateLimitError as e:
        #         attempt += 1
        #         delay = min(initial_delay * (2 ** attempt) * (1 + random.uniform(0, 0.1)), max_delay)
        #         self.logger.warning(f"Rate limit exceeded on attempt {attempt}. Retrying in {delay:.2f} seconds... ({e})")
        #         time.sleep(delay)
        #     except Exception as e:
        #         self.logger.error(f"Unexpected error on attempt {attempt}: {e}")
        #         raise  # Raise other exceptions immediately

        # raise Exception("Exceeded maximum retries for embedding documents due to rate limit.")
