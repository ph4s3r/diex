import sys
import time
import logging

import pickle
import voyageai
from voyageai.error import InvalidRequestError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError
    )
from langchain_core.documents import Document

from components.interfaces.all_interfaces import Embedder

class EmbedderAPIError(Exception):
    """Custom exception for embedder API-related errors."""
    pass

def validate_vector(vec, logger):
    try:
        if not isinstance(vec, list):
            raise EmbedderAPIError("Non-list vector format received from embedder API. Please review.")
        if len(vec) != 1:
            raise EmbedderAPIError("Non-1 element vector list received from embedder API. Please review.")
    except EmbedderAPIError as e:
        logger.error(str(e))
        raise

class VoyageEmbedder(Embedder):
    def __init__(self,
                 output_dimension,
                 output_dtype,
                 model,
                 batch_size):
        
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        self.model = model
        self.batch_size = batch_size
        self.vo = voyageai.Client()
        self.logger = logging.getLogger('Embedder')

    def conntest(self, max_retries: int = 4):
        """
        Basic connectivity test for the Voyage AI embedding endpoint.
        Tries to embed a simple "hello world" string with the model and
        parameters already configured on this instance.
        """

        retries = 0
        while True:
            try:
                self.vo.embed(
                    texts=["hello world"],
                    model=self.model,
                    input_type="document",
                    output_dimension=self.output_dimension,
                    output_dtype=self.output_dtype,
                    truncation=True,
                )
                self.logger.info(f"Established connection with Voyage AI (model: {self.model}).")
                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    self.logger.error(f"Voyage AI connection test failed after {retries} attempts: {e}")
                    sys.exit(1)
                self.logger.warning(f"Voyage AI connection test failed ({e}); retrying in 3 seconds…")
                time.sleep(3)


    def embed(self, documents: list["Document"]) -> list[list[float]]:
        self.logger.info("Embedding starts with Voyage EMBEDDING API in batches")
        batch_size = self.batch_size  # star using max allowed, to reduce RPM, then if we hit TPM, lower it
        all_embeddings = []
        failed_batches = []

        texts = [doc.page_content for doc in documents]
        
        # tenacity exponential backoff per batch
        @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
        def embed_batch(texts_batch: list[str]):
            return self.vo.embed(
                texts=texts_batch,
                model=self.model,           
                input_type="document",      
                output_dimension=self.output_dimension,      
                output_dtype=self.output_dtype,
                truncation=True             # if this happens... voyage3 models have 32k token limit...
            )
        
        for i in range(0, len(texts), batch_size):
            texts_batch = texts[i:i + batch_size]
            self.logger.info(f"Embedding batch {i // batch_size + 1} with {len(texts_batch)} documents")
            try:
                result = embed_batch(texts_batch)
                print("DEBUG")
            except RetryError as re:
            # oversize batch error sample: 
            # voyageai.error.InvalidRequestError: Request to model 'voyage-3-large' failed. 
            # The max allowed tokens per submitted batch is 120000. 
            # Your batch has 142456 tokens after truncation. 
            # Please lower the number of tokens in the batch.
            # Solution: need to unwrap InvalidRequestError from tenacity RetryError
                exc = re.last_attempt.exception()          
                if isinstance(exc, InvalidRequestError) and \
                "max allowed tokens per submitted batch" in str(exc):
                    # so we just split it into 2 batches, should be fine
                    print("split a batch that was too big into 2...")
                    result = embed_batch(texts_batch[:64])
                    self.logger.info(f"Re-embedding batch {i // batch_size + 1} with {len(texts_batch)} documents")
                    result2 = embed_batch(texts_batch[64:])
                    self.logger.info(f"Re-embedding batch {i // batch_size + 1} with {len(texts_batch)} documents")
                    result.embeddings.extend(result2.embeddings)
            except Exception as e:
                self.logger.error("Embedding batch failed: " + str(e))
                failed_batches.append(texts_batch)
                continue

            # The API returns embeddings in a list, iterate and validate each.
            for vec in result.embeddings:
                try:
                    validate_vector([vec], self.logger)
                    all_embeddings.append(vec)
                except EmbedderAPIError:
                    sys.exit(100)

        if len(failed_batches) > 0:
            self.logger.warning("you can find failed batches in the file voyage_failed_batches in the root dir")
            with open("voyage_failed_batches", "wb") as f:
                pickle.dump(failed_batches, f)

        return all_embeddings
