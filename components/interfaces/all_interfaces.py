from abc import ABC, abstractmethod
from typing import List, Any

from langchain_core.documents import Document


class DocumentLoader(ABC):
    @abstractmethod
    # load data from a document source and return them in a List of langchain_core.documents.base.Document type
    def load(self, path: str) -> List[Document]:
        pass


class DocumentSplitter(ABC):
    @abstractmethod
    # split langchain_core.documents.base.Document items to smaller chunks
    def split(self, documents: List[Document]) -> List[Document]:
        pass


class Embedder(ABC):
    @abstractmethod
    # create vector embeddings from langchain_core.documents.base.Document items
    def embed(self, documents: List[Document]) -> Any:
        pass


class VectorInserter(ABC):
    @abstractmethod
    # insert vector embeddings to a vector database
    def upload(self, embeddings: Any) -> str:
        pass
