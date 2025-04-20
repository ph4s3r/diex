import logging

from typing import List
from langchain_core.documents import Document

from components.interfaces.all_interfaces import SemanticChunker


class UnstructuredHTMLChunker(SemanticChunker):

    def __init__(self, ) -> None:
        
        self.logger: logging.Logger = logging.getLogger('SemanticChunker')


    def chunk(self, documents: List[Document]) -> List[Document]:

        return documents
    

class UnstructuredMarkdownChunker(SemanticChunker):

    def __init__(self) -> None:
        
        self.logger: logging.Logger = logging.getLogger('SemanticChunker')


    def chunk(self, documents: List[Document]) -> List[Document]:

        return documents