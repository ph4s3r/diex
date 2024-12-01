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

        self.api_key = "pcsk_r1c8Y_HvcdmYqVuYkhCLtjLYBTzGbDSYpJn3Mxf47SCkmXHBGUHLF2n6r9VcswV4gHzos"
        self.index_name = "universe"
        self.index_host = f"https://{self.index_name}-226a147.svc.aped-4627-b74a.pinecone.io"
        self.namespace = "ns1"
        self.logger = logging.getLogger('Inserter')
        

        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(host=self.index_host)
        self.logger.info(f"Established connection with index at host {self.index_host}, index name: {self.index_name}")

    def _batchify(self, items: List[dict], batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

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

        for doc, vec in zip(documents, vectors):
            embeddings.append({
                "id": doc.id,
                "values": vec,
                "metadata": {
                    **doc.metadata,
                    "content": doc.page_content  # Add the text content as metadata with the key "content"
                }
            })

        # Split embeddings into batches
        for batch_num, batch in enumerate(self._batchify(embeddings, self.MAX_BATCH_SIZE), start=1):
            try:
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace
                )
                self.logger.info(f"Successfully upserted batch {batch_num} with {len(batch)} vectors.")
            except Exception as e:
                self.logger.error(f"Failed to upsert batch {batch_num}: {e}")

        self.logger.info("Successfully upserted the stuff YaYY! :)")


