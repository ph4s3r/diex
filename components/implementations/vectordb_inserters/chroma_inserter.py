from components.interfaces.all_interfaces import VectorInserter

import os
import logging
import chromadb
from uuid import uuid4
from typing import List, Optional
from langchain_core.documents import Document
import chromadb.utils.embedding_functions as embedding_functions

# to avoid messing with a python image that does not have the required sqlite
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


class ChromaDBRemoteInserter(VectorInserter):
    def __init__(self,
                 url: str,
                 port: int,
                 tenant: str,
                 database: str,
                 collection: str,
                 embedding_model: str
    ) -> None:

        self.url = url
        self.port= port
        self.tenant= tenant
        self.database= database
        self.collection= collection
        self.embedding_model= embedding_model
    
        self.logger = logging.getLogger('Inserter')

    def insert(self, documents: List[Document], vectors: Optional[List[List[float]]] = None) -> None:

        self.logger.info("Starting Inserting process to ChromaDB")
        self.logger.info(f"Will use {str(self.embedding_model)} embedding model")
        self.logger.info(f"Initiating chroma connection at {self.url}:{self.port}, tenant: {self.tenant}, database: {self.database}")

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=os.getenv("OPENAI_API_KEY"),
                        model_name=self.embedding_model
                    )

        try:
            self.chroma_client: chromadb.ClientAPI = chromadb.HttpClient(
                host=self.url,
                port=self.port
                )
        except ValueError as e:
            self.logger.error(f"{e}")
            os._exit(100)


        self.logger.info("Trying to get collections")
        collections: os.Sequence[chromadb.Collection] = self.chroma_client.list_collections()
        if len(collections) > 0:
            self.logger.info(f"Collections found: {collections}")
        else:
            self.logger.info(f"No collections found, but will create one!")

        collection = self.chroma_client.get_or_create_collection(
            name = self.collection,
            embedding_function = openai_ef
            )
        
        self.logger.info(f"Collection {self.collection} found")

        uuids = [str(uuid4()) for _ in range(len(documents))]

        documents_str = [document.page_content for document in documents]

        if vectors is None:
            self.logger.info("No vectors received, will let langchain to do the embeddings.")
            self.logger.info("Trying to upsert the documents to the collection")
            collection.upsert(
                documents=documents_str,
                ids=uuids
            )
        else:
            self.logger.info("Trying to upsert the vectors and documents to the collection")
            metadatas = [document.metadata for document in documents]
            collection.upsert(
                documents=documents_str,
                embeddings=vectors,
                metadatas=metadatas,
                ids=uuids
            )

        self.logger.info("Successfully upserted all the stuff YaYY! :)")






    # def list_indexed_documents(self) -> List[Tuple[dict, str]]:
    #     """
    #     Lists all indexed documents, returning a list of tuples with metadata and content.
    #     """

    #     results = self.chroma_client._collection.get() 
        
    #     documents = []
    #     for metadata, content in zip(results["metadatas"], results["documents"]):
    #         documents.append((metadata, content))
        
    #     return documents