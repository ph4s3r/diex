from abc import ABC, abstractmethod

from langchain_core.documents import Document


class DocumentLoader(ABC):
    @abstractmethod
    # load data from a document source and return them in a list of langchain_core.documents.base.Document type per page
    def load(self) -> list[Document]:
        pass

class SemanticChunker(ABC):
    @abstractmethod
    # chunk doc pages semantically to smaller chunks
    def chunk(self, documents: list[Document]) -> list[Document]:
        pass


class DocumentSplitter(ABC):
    @abstractmethod
    # split docs based on token length
    def split(self, documents: list[Document]) -> list[Document]:
        pass


class Embedder(ABC):
    @abstractmethod
    # create vector embeddings from langchain_core.documents.base.Document items
    def embed(self, documents: list[Document]) -> list[list[float]]:
        pass


class VectorInserter(ABC):
    @abstractmethod
    # insert vector embedding - doc pairs to a vector database
    def insert(self, documents: list[Document], vectors: list[list[float]])-> None:
        pass
