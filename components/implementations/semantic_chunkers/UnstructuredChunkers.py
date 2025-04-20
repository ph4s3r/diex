import logging
import json

from typing import List
from langchain_core.documents import Document

from components.interfaces.all_interfaces import SemanticChunker


class UnstructuredHTMLChunker(SemanticChunker):

    def __init__(self, ) -> None:
        
        self.logger: logging.Logger = logging.getLogger('SemanticChunker')


    def chunk(self, documents: List[Document]) -> List[Document]:

        docs = documents

        if self.debug:
            saveout = [
                {"content-length": len(d.page_content), "content": d.page_content}
                for d in docs
            ]
            with open("parsed-pdf-chunks.json", "w", encoding="utf-8") as f:
                json.dump(saveout, f, indent=2, ensure_ascii=False)

        return documents
    

class UnstructuredMarkdownChunker(SemanticChunker):

    def __init__(self) -> None:
        
        self.debug = False
        self.logger: logging.Logger = logging.getLogger('SemanticChunker')


    def chunk(self, documents: List[Document]) -> List[Document]:

        docs = documents

        if self.debug:
            saveout = [
                {"content-length": len(d.page_content), "content": d.page_content}
                for d in docs
            ]
            with open("parsed-md-chunks.json", "w", encoding="utf-8") as f:
                json.dump(saveout, f, indent=2, ensure_ascii=False)

        return documents