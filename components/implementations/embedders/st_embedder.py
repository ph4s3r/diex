import os
import logging
from typing import List

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
            # self.logger.info("Max Sequence Length:", self.model.max_seq_length, "changing it to ")
            self.model.max_seq_length = 512
            # self.model.tokenizer.padding_side = "right"

        except Exception as e:
            self.logger.error(f"Failed to load {self.embedding_model} model: {e}")
            raise


    def embed(self, documents: List[Document]) -> List[List[float]]:
        if not documents:
            self.logger.error("No documents provided for embedding.")
            os._exit(199)

        batch_size = 2500
        embedding_vectors = list()

        batches = (len(documents) // batch_size) + 1
            
        for i in range(batches):
            range_start = i*batch_size
            range_end = i*batch_size+batch_size

            dox = [doc.page_content for doc in documents[range_start:range_end]]
            
            embeddings_nparray = self.model.encode(dox)
            self.logger.info(f"docs embedded successfully, embeddings type: {type(embeddings_nparray)} shape: {embeddings_nparray.shape}")
            embedding_vectors.extend(embeddings_nparray.tolist())

        return embedding_vectors



        # code for multi-cpu or multi-gpu training:

        # Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
        # if __name__ == "__main__":
        #     # Create a large list of 100k sentences
        #     sentences = [f"This is sentence {i}" for i in range(100000)]

        #     # Define the model
        #     model = SentenceTransformer("all-MiniLM-L6-v2")

        #     # Start the multi-process pool on all available CUDA devices
        #     pool = model.start_multi_process_pool()

        #     # Compute the embeddings using the multi-process pool
        #     emb = model.encode_multi_process(sentences, pool)
        #     print("Embeddings computed. Shape:", emb.shape)

        #     # Optional: Stop the processes in the pool
        #     model.stop_multi_process_pool(pool)


