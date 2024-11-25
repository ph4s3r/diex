import os
import logging
from typing import List

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from components.interfaces.all_interfaces import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self):
        """
        Sentence Transformer Embedder
        
        **Sentence Transformers:**
        - [Sentence Transformers](https://sbert.net/)
        - [MTEB Leaderboards](https://huggingface.co/spaces/mteb/leaderboard)

        **NV-Embed-v2:**
        - [NV-Embed Paper](https://arxiv.org/pdf/2405.17428)
        - [NV-Embed-v2 on Hugging Face](https://huggingface.co/nvidia/NV-Embed-v2)
        - [Other Retrieval Models](https://build.nvidia.com/explore/retrieval)
        - [MTEB Leaderboards](https://huggingface.co/spaces/mteb/leaderboard)

        **NV-Embed-v2 requirements:**
        - The model requires approximately 30GB of GPU memory.
        - Ensure you have access to the NVIDIA NV-Embed-v2 model on Hugging Face.
        """
        self.logger = logging.getLogger('Embedder')

        # load model
        try:
            self.model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
            self.logger.info("NV-Embed-v2 model loaded successfully.")
            self.logger.info("Max Sequence Length:", self.model.max_seq_length, "changing it to ", 32768)
            self.model.max_seq_length = 32768
            self.model.tokenizer.padding_side = "right"

        except Exception as e:
            self.logger.error(f"Failed to load NV-Embed-v2 model: {e}")
            raise


    def embed(self, documents: List[Document]) -> List[List[float]]:
        if not documents:
            self.logger.error("No documents provided for embedding.")
            os._exit(199)

        texts = [doc.page_content for doc in documents]

        def add_eos(input_examples):
            input_examples = [input_example + self.model.tokenizer.eos_token for input_example in input_examples]
            return input_examples
        
        batch_size = 1
        embeddings = self.model.encode(add_eos(texts), batch_size=batch_size, normalize_embeddings=True)
        self.logger.info(f"docs embedded successfully, embeddings type: {type(embeddings)} shape: {embeddings.shape}")

        return embeddings



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


