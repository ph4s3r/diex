from components.interfaces.all_interfaces import DocumentSplitter

import logging
from typing import List
from uuid import uuid3, NAMESPACE_DNS
from langchain_core.documents import Document
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TransformersSplitter(DocumentSplitter):
    def __init__(self, 
                 max_token_seq_len, 
                 token_overlap
                 ):
        
        self.max_token_seq_len = max_token_seq_len
        self.token_overlap = token_overlap

        # giving 33% smaller token length since this splitter is sometimes massively overshooting
        langchain_splitter_token_max_len = self.max_token_seq_len - self.max_token_seq_len/3
        # this is unsafe though... would be better to manually recursively split
        self.langchain_recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=langchain_splitter_token_max_len, 
            chunk_overlap=self.token_overlap
        )
        model_dir = "models/stella_en_1.5B_v5"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        self.logger = logging.getLogger('Splitter')

    def split(self, documents: List[Document]) -> List[Document]:
        """Splits markdown documents into smaller chunks based on headers and character limits."""
        all_chunks = []

        for doc in documents:
            # chunking based on the model's innate sequence length
            input_data = self.tokenizer(doc.page_content, padding="longest", truncation=False, max_length=self.max_token_seq_len, return_tensors="pt")
            seq_length = len(input_data.encodings[0].tokens)
            if seq_length > 512:
                # self.logger.info(f"splitting doc because seq_length > 512. seq len = {seq_length}")
                chunked_docs = self.chunk(doc)
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

