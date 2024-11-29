from components.interfaces.all_interfaces import DocumentSplitter

from uuid import uuid3, NAMESPACE_DNS
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

class MDSplitter(DocumentSplitter):
    def __init__(self, 
                 chunk_size, 
                 chunk_overlap, 
                 headers_to_split_on):
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
            ]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.headers_to_split_on = headers_to_split_on

    def split(self, documents: List[Document]) -> List[Document]:
        """Splits markdown documents into smaller chunks based on headers and character limits."""
        all_chunks = []
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on, strip_headers=False
        )

        for doc in documents:
            md_header_splits = markdown_splitter.split_text(doc.page_content)
            for split in md_header_splits:
                last_header = split.page_content.split("\n")[0] # extract header
                split.metadata = doc.metadata

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                
                # Attach header metadata to each chunk
                for chunk in text_splitter.split_text(split.page_content):
                    chunk_doc = Document(
                        id = str(uuid3(NAMESPACE_DNS, chunk)),
                        page_content=chunk, 
                        metadata={**split.metadata, "header": last_header}
                    )
                    all_chunks.append(chunk_doc)

        return all_chunks
