from components.interfaces.all_interfaces import DocumentSplitter

import logging
from typing import List
from uuid import uuid3, NAMESPACE_DNS
from langchain_core.documents import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class MDTokenSplitter(DocumentSplitter):
    def __init__(self, 
                 max_token_seq_len, 
                 token_overlap, 
                 headers_to_split_on):
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
            ]
        self.max_token_seq_len = max_token_seq_len
        self.token_overlap = token_overlap
        self.headers_to_split_on = headers_to_split_on

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",chunk_size=self.max_token_seq_len, chunk_overlap=self.token_overlap
        )

        self.logger = logging.getLogger('Splitter')

    def split(self, documents: List[Document]) -> List[Document]:
        """Splits markdown documents into smaller chunks based on headers and character limits."""
        all_chunks = []
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on, strip_headers=False
        )

        for doc in documents:
            md_header_splits = markdown_splitter.split_text(doc.page_content)
            self.logger.info(f"Markdown split to {len(md_header_splits)} splits.")
            for split in md_header_splits:
                last_header = split.page_content.split("\n")[0] # extract header
                split.metadata = doc.metadata
                
                # Attach header metadata to each chunk
                for i, chunk in enumerate(self.text_splitter.split_text(split.page_content)):
                    chunk_doc = Document(
                        id = str(uuid3(NAMESPACE_DNS, chunk)),
                        page_content=chunk, 
                        metadata={**split.metadata, "header": last_header}
                    )
                    all_chunks.append(chunk_doc)
            self.logger.info(f"The md split is split to {i} chunks.")

        return all_chunks
