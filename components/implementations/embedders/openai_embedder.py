from components.interfaces.all_interfaces import Embedder

from openai import RateLimitError
import logging
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Import Tenacity for retrying with exponential backoff
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log
)


# Import tqdm for progress bar
from tqdm import tqdm


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model: str, 
        batch_size: int
    ):
        """
        Initializes the OpenAIEmbedder with the specified model and batch size.
        https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document
        https://python.langchain.com/docs/integrations/document_loaders/unstructured_markdown/
        https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html

        Args:
            model (str): The OpenAI model to use for embeddings.
            batch_size (int): The number of documents to embed in each batch.
        """
        self.model = model
        self.batch_size = batch_size

        self.logger = logging.getLogger('Embedder')

    def embed(self, documents: List["Document"]) -> List[List[float]]:
        """
        Embeds the provided documents using OpenAI's Embeddings API with rate limiting.

        Implements exponential backoff to handle RateLimitError exceptions gracefully.

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

        # Static rate limiting configuration
        max_retries = 5          # Maximum number of retry attempts
        initial_delay = 1.0      # Initial delay between retries in seconds
        max_delay = 60.0         # Maximum delay between retries in seconds

        model = str(self.model)

        embeddings = OpenAIEmbeddings(model=model)

        @retry(
            reraise=True,
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=initial_delay, max=max_delay),
            stop=stop_after_attempt(max_retries),
            before_sleep=before_sleep_log(self.logger, logging.WARNING)
        )
        def get_embeddings(batch_texts: List[str]) -> List[List[float]]:
            """
            Retrieves embeddings for the provided batch of texts.

            Args:
                batch_texts (List[str]): A list of text strings to embed.

            Returns:
                List[List[float]]: A list of embedding vectors.
            """
            return embeddings.embed_documents(batch_texts)

        all_vectors = []

        # Split texts into batches based on batch_size
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        total_batches = len(batches)

        # Initialize the progress bar
        with tqdm(total=total_batches, desc="Embedding documents", unit="batch") as pbar:
            for batch in batches:
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
