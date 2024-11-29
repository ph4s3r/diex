from components.interfaces.all_interfaces import VectorInserter

import time
import logging
from typing import List, Optional
from pinecone import Pinecone
from langchain_core.documents import Document


class PineConeUpserter(VectorInserter):

    MAX_BATCH_SIZE = 500

    def __init__(self
    ) -> None:

        self.api_key = "PINECONE_API_KEY_REMOVED"
        self.index_host = "https://test-226a147.svc.aped-4627-b74a.pinecone.io"
        self.index_name = "test"
        self.namespace = "ns1"
        self.logger = logging.getLogger('Inserter')
        

        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(host=self.index_host)
        self.logger.info(f"Established connection with index at host {self.index_host}, index name: {self.index_name}")

    def insert(self, documents: List[Document], vectors: Optional[List[List[float]]] = None) -> None:

        if vectors is None:
            self.logger.error(f"No vectors received by PineCone Upserter, returning.")
            return None
        
        self.logger.info(f"Inserting will be attempted to namespace: {self.namespace}")

        # how to do this with host? really it is just index_name?
        # https://github.com/pinecone-io/pinecone-python-client/blob/main/pinecone/control/pinecone.py
        # https://sdk.pinecone.io/python/pinecone.html#describe-index
        
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)

        embeddings = []

        for d, e in zip(documents, vectors):
            embeddings.append({
                "id": d.id,
                "values": e,
                "metadata": d.metadata
            })

        self.index.upsert(
            vectors=embeddings,
            namespace=self.namespace
        )

        self.logger.info("Successfully upserted the stuff YaYY! :)")


