from components.interfaces.all_interfaces import VectorInserter

import chromadb
import logging
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings


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
        self.port = port
        self.tenant=tenant
        self.database=database
        self.collection=collection
        self.embedding_function: chromadb.EmbeddingFunction = OpenAIEmbeddings(model=embedding_model)
        self.chroma_client: chromadb.ClientAPI = chromadb.HttpClient(
            host=self.host, 
            port=self.port,
            tenant=self.tenant,
            database=self.database
            )

        self.logger = logging.getLogger('Inserter')

    def insert(self, vectors: List[List[float]]) -> None:

        self.logger.info("Starting Inserting process to ChromaDB")
        self.logger.info(f"Received {len(vectors)} vectors")
        self.logger.info(f"Chroma connection: {self.url}:{self.port}, tenant: {self.tenant}, database: {self.database}")
        self.logger.info(f"Will use {str(self.embedding_fmodel)} embedding model")

        collections = self.chroma_client.list_collections()
        if len(collections) > 0:
            self.logger.info(f"Collections found: {collections}")
        else:
            self.logger.info(f"No collections found, but will create one!")

        collection = self.chroma_client.get_or_create_collection(
            name = self.collection,
            embedding_function = self.embedding_function
            )

    

    # def list_indexed_documents(self) -> List[Tuple[dict, str]]:
    #     """
    #     Lists all indexed documents, returning a list of tuples with metadata and content.
    #     """

    #     results = self.chroma_client._collection.get() 
        
    #     documents = []
    #     for metadata, content in zip(results["metadatas"], results["documents"]):
    #         documents.append((metadata, content))
        
    #     return documents