from components.interfaces.all_interfaces import VectorInserter

import sys
import time
import logging
from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import Pinecone
from langchain_core.documents import Document


class PineConeInserter(VectorInserter):

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

        

    def conntest(self, max_retries: int = 4):

        retried = 0

        while not self.pc.describe_index(self.index_name).status['ready']:
            self.logger.warning("PineCone Index unavailable at the moment, retrying in every 3 seconds")
            time.sleep(3)
            retried = retried + 1
            if retried == max_retries:
                self.logger.error("PineCone Index is unavailable, exiting.")
                sys.exit(1)
        self.logger.info(f"Established connection with index at host {self.index_host}, index name: {self.index_name}")

    def _batchify(self, items: list[dict], batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def insert(self, documents: list[Document], vectors: list[list[float]] = None) -> None:

        self.conntest()
        
        # check the whole thing again for vector containers that does not have real value - where the embedding has basically failed

        nullvectors = [] # gather the indexes here
        for i, v in enumerate(vectors):
            if v[0] is None:
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
            time.sleep(3)

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


