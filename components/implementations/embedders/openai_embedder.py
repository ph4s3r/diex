from components.interfaces.all_interfaces import Embedder
from openai import RateLimitError
import logging
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log
)
from tqdm import tqdm
import tiktoken
import time
import threading


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model: str, 
        batch_size: int,
        max_tokens_per_minute: int,
        max_requests_per_minute: int,
        batch_queue_limit: int
    ):
        """
        Initializes the OpenAIEmbedder with the specified model and rate limits.

        Args:
            model (str): The OpenAI model to use for embeddings.
            batch_size (int): The maximum number of documents to embed in each batch.
            max_tokens_per_minute (int): Maximum number of tokens allowed per minute.
            max_requests_per_minute (int): Maximum number of API requests allowed per minute.
            batch_queue_limit (int): Maximum number of tokens allowed in the batch queue.
        """
        self.model = model
        self.batch_size = batch_size
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_minute = max_requests_per_minute
        self.batch_queue_limit = batch_queue_limit

        # Configure logger
        self.logger = logging.getLogger('Embedder')

        # Initialize tokenizer using the appropriate encoding for the model
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.logger.warning(f"Model {self.model} not recognized. Falling back to 'cl100k_base' encoding.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize rate limiting counters
        self.lock = threading.Lock()
        self.reset_time_minute = time.time() + 60  # Reset every minute
        self.requests_made_minute = 0
        self.tokens_used_minute = 0
        self.current_queue_tokens = 0

    def _count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a given text."""
        return len(self.tokenizer.encode(text))

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Splits texts into batches based on both document count and token limits.

        Args:
            texts (List[str]): List of text strings to batch.

        Returns:
            List[List[str]]: List of text batches.
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            tokens = self._count_tokens(text)
            
            # Skip documents that individually exceed the max_tokens_per_minute
            if tokens > self.max_tokens_per_minute:
                self.logger.warning(f"Single document exceeds max tokens per minute ({tokens} tokens). Skipping document.")
                continue

            # Check if adding the current document exceeds either the batch size or token limits
            if (len(current_batch) + 1 > self.batch_size) or (current_tokens + tokens > self.max_tokens_per_minute):
                if current_batch:
                    # Check batch queue limit before adding
                    batch_total_tokens = current_tokens
                    if self.current_queue_tokens + batch_total_tokens > self.batch_queue_limit:
                        self.logger.warning(f"Batch queue limit reached. Skipping batch of {batch_total_tokens} tokens.")
                        current_batch = []
                        current_tokens = 0
                        continue
                    self.logger.info(f"Creating batch with {len(current_batch)} documents and {current_tokens} tokens.")
                    batches.append(current_batch)
                    self.current_queue_tokens += batch_total_tokens
                # Start a new batch
                current_batch = [text]
                current_tokens = tokens
            else:
                current_batch.append(text)
                current_tokens += tokens

        # Add the last batch if it exists and doesn't exceed the queue limit
        if current_batch:
            batch_total_tokens = current_tokens
            if (self.current_queue_tokens + batch_total_tokens <= self.batch_queue_limit):
                self.logger.info(f"Creating batch with {len(current_batch)} documents and {current_tokens} tokens.")
                batches.append(current_batch)
                self.current_queue_tokens += batch_total_tokens
            else:
                self.logger.warning(f"Batch queue limit reached. Skipping batch of {batch_total_tokens} tokens.")

        self.logger.info(f"Total batches created: {len(batches)}")
        return batches

    def _rate_limit(self, tokens: int):
        """
        Ensures that the total tokens and requests used do not exceed the limits.

        Args:
            tokens (int): Number of tokens in the current request.
        """
        with self.lock:
            current_time = time.time()

            # Reset minute counters if needed
            if current_time >= self.reset_time_minute:
                self.requests_made_minute = 0
                self.tokens_used_minute = 0
                self.reset_time_minute = current_time + 60

            # Check RPM (Requests Per Minute)
            if self.requests_made_minute + 1 > self.max_requests_per_minute:
                sleep_duration = self.reset_time_minute - current_time
                self.logger.info(f"RPM limit reached. Sleeping for {sleep_duration:.2f} seconds.")
                time.sleep(sleep_duration)
                # Reset after sleeping
                self.requests_made_minute = 0
                self.tokens_used_minute = 0
                self.reset_time_minute = time.time() + 60

            # Check TPM (Tokens Per Minute)
            if self.tokens_used_minute + tokens > self.max_tokens_per_minute:
                sleep_duration = self.reset_time_minute - current_time
                self.logger.info(f"TPM limit reached. Sleeping for {sleep_duration:.2f} seconds.")
                time.sleep(sleep_duration)
                # Reset after sleeping
                self.requests_made_minute = 0
                self.tokens_used_minute = 0
                self.reset_time_minute = time.time() + 60

            # Update counters
            self.requests_made_minute += 1
            self.tokens_used_minute += tokens

    def embed(self, documents: List["Document"]) -> List[List[float]]:
        """
        Embeds the provided documents using OpenAI's Embeddings API with rate limiting.

        Args:
            documents (List[Document]): A list of documents to embed.

        Returns:
            List[List[float]]: A list of embedding vectors corresponding to the documents.
        """
        texts = []

        self.logger.info("Embedding starts")
        try:
            for doc in documents:
                texts.append(doc.page_content)
        except Exception as e:
            self.logger.error(f"Cannot get content from documents: {e}")
            raise

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model=self.model)

        @retry(
            reraise=True,
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=20, max=60),
            stop=stop_after_attempt(10),
            before_sleep=before_sleep_log(self.logger, logging.WARNING)
        )
        def get_embeddings(batch_texts: List[str]) -> List[List[float]]:
            return embeddings.embed_documents(batch_texts)

        # Create token-based batches
        batches = self._create_batches(texts)
        total_batches = len(batches)

        all_vectors = []

        # Initialize the progress bar
        with tqdm(total=total_batches, desc="Embedding documents", unit="batch") as pbar:
            for batch in batches:
                batch_tokens = sum(self._count_tokens(text) for text in batch)
                batch_documents = len(batch)

                # Enforce rate limiting before making the API call
                self._rate_limit(batch_tokens)

                # Log the number of tokens and documents in the current batch
                self.logger.info(f"Embedding batch: {batch_documents} documents, {batch_tokens} tokens")

                try:
                    vectors = get_embeddings(batch)
                    if vectors:
                        all_vectors.extend(vectors)
                except RateLimitError as e:
                    self.logger.error(f"Exceeded maximum retries due to rate limit. Last error: {e}")
                    raise
                except Exception as e:
                    self.logger.error(f"Unexpected error during embedding: {e}")
                    raise
                finally:
                    pbar.update(1)

        self.logger.info(f"Documents got embedded to {len(all_vectors)} vectors")
        return all_vectors
