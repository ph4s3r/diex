from components.interfaces.all_interfaces import DocumentSplitter

import logging
from typing import List
from uuid import uuid3, NAMESPACE_DNS
from langchain_core.documents import Document
from transformers import AutoTokenizer
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class TransformersSplitter(DocumentSplitter):
    def __init__(self, 
                 max_token_seq_len, 
                 token_overlap
                 ):
        
        self.max_token_seq_len = max_token_seq_len
        self.token_overlap = token_overlap

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",chunk_size=self.max_token_seq_len, chunk_overlap=self.token_overlap
        )
        model_dir = "models/stella_en_1.5B_v5"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        self.logger = logging.getLogger('Splitter')

    def split(self, documents: List[Document]) -> List[Document]:
        """Splits markdown documents into smaller chunks based on headers and character limits."""
        all_chunks = []

        for doc in documents:            
            # Attach header metadata to each chunk
            for i, chunk in enumerate(self.text_splitter.split_text(doc.page_content)):
                input_data = self.tokenizer(chunk, padding="longest", truncation=True, max_length=self.max_token_seq_len, return_tensors="pt")
                self.logger.debug(f"char len of chunk is {len(doc.page_content)}")
                self.logger.debug(f"seq len of chunk is {len(input_data.encodings[0].tokens)}")
                if input_data.n_sequences > 1:
                    print("would need to split here!")
                chunk_doc = Document(
                    id = str(uuid3(NAMESPACE_DNS, chunk)),
                    page_content=chunk, 
                    metadata=doc.metadata
                )
                all_chunks.append(chunk_doc)

        return all_chunks
