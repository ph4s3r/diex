from components.interfaces.all_interfaces import DocumentSplitter

import logging
from typing import List
from uuid import uuid3, NAMESPACE_DNS
from langchain_core.documents import Document
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VoyageSplitter(DocumentSplitter):
    def __init__(self, 
                 max_token_seq_len, 
                 token_overlap
                 ):
        
        self.max_token_seq_len = max_token_seq_len
        self.token_overlap = token_overlap

        # giving much smaller token length since this splitter is sometimes massively undershooting
        langchain_splitter_token_max_len = self.max_token_seq_len/3
        # this is unsafe though... would be better to manually recursively split
        self.langchain_recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=langchain_splitter_token_max_len, 
            chunk_overlap=self.token_overlap
        )
        self.tokenizer = AutoTokenizer.from_pretrained("voyageai/voyage-3")

        self.logger = logging.getLogger('Splitter')

    def split(self, documents: List[Document]) -> List[Document]:
        """Splits markdown documents into smaller chunks based on headers and character limits."""
        all_chunks = []

        for doc in documents:
            # chunking based on the model's sequence length calculation
            # need to change this to recursive, because now we call it just twice... unmaintainable code
            if len(self.tokenizer.tokenize(doc.page_content)) > self.max_token_seq_len:
                chunked_docs = self.chunk(doc)
                for i, cd in enumerate(chunked_docs):
                    if len(self.tokenizer(cd.page_content, padding="longest", truncation=False, max_length=self.max_token_seq_len, return_tensors="pt").encodings[0].tokens) > self.max_token_seq_len:
                        chunked_docs_level2 = self.chunk(cd)
                        all_chunks.extend(chunked_docs_level2)
                        chunked_docs.pop(i)
                all_chunks.extend(chunked_docs)
            else:
                chunk_doc = Document(
                    id = str(uuid3(NAMESPACE_DNS, doc.page_content)),
                    page_content=doc.page_content, 
                    metadata=doc.metadata
                )
                all_chunks.append(chunk_doc)

        return all_chunks
    
    def chunk(self, doc: Document):

        allchunks = []
        
        chunks = self.langchain_recursive_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunk_doc = Document(
                id = str(uuid3(NAMESPACE_DNS, chunk)),
                page_content=chunk, 
                metadata=doc.metadata
            )
            allchunks.append(chunk_doc)

        return allchunks

