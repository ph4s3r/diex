from components.interfaces.all_interfaces import VectorInserter

import time
import logging
from typing import List
from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import Pinecone
from langchain_core.documents import Document


class PineConeUpserter(VectorInserter):

    MAX_BATCH_SIZE = 300

    def __init__(self,
                 api_key,
                 index_name,
                 namespace,
                 index_host_suffix

    ) -> None:

        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        
        self.index_host = f"https://{index_name}-{index_host_suffix}"
        self.logger = logging.getLogger('Inserter')
        
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(host=self.index_host)
        self.logger.info(f"Established connection with index at host {self.index_host}, index name: {self.index_name}")

    def _batchify(self, items: List[dict], batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def insert(self, documents: List[Document], vectors: List[List[float]] = None) -> None:

        if vectors is None:
            self.logger.error(f"No vectors received by PineCone Upserter, returning.")
            return None
        
        # check the whole thing again for vector containers that does not have real value - where the embedding has basically failed

        nullvectors = [] # gather the indexes here
        for i, v in enumerate(vectors):
            if v[0] == None:
                nullvectors.append(i)

        if len(nullvectors) > 0:
            self.logger.info(f"Need to remove {len(nullvectors)} empty vector embeddings from the list:")
            for index_of_nullvector in reversed(nullvectors):
                self.logger.info(f"Removing: {documents[index_of_nullvector]}")
                documents.pop(index_of_nullvector)
                vectors.pop(index_of_nullvector)
            

        
        self.logger.info(f"Inserting will be attempted to namespace: {self.namespace}")

        # how to do this with host? really it is just index_name?
        # https://github.com/pinecone-io/pinecone-python-client/blob/main/pinecone/control/pinecone.py
        # https://sdk.pinecone.io/python/pinecone.html#describe-index
        
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)

        embeddings = []

        for doc, vec in zip(documents, vectors):
            meta = doc.metadata
            meta.update({"content": doc.page_content})
            embeddings.append({
                "id": doc.id,
                "values": vec,
                "metadata": meta
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


