from components.interfaces.all_interfaces import DocumentSplitter

from typing import List
from uuid import uuid3, NAMESPACE_DNS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TiktokenRecursiveSplitter(DocumentSplitter):
    def __init__(self, 
                 max_token_seq_len, 
                 token_overlap
                 ):
        
        self.max_token_seq_len = max_token_seq_len
        self.token_overlap = token_overlap

        # for details of this token-based splitting see https://python.langchain.com/docs/how_to/split_by_token/
        self.langchain_recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=max_token_seq_len, 
            chunk_overlap=self.token_overlap
        )


    def split(self, documents: List[Document]) -> List[Document]:
        """Splits documents by tokens with LangChain's RecursiveCharacterTextSplitter using tiktoken
        """

        allsplits = []

        for doc in documents:
            splits = self.langchain_recursive_splitter.split_text(doc.page_content)
            for split in splits:
                split_doc = Document(
                    id = str(uuid3(NAMESPACE_DNS, split)),
                    page_content=split, 
                    metadata=doc.metadata
                )
                allsplits.append(split_doc)

        return allsplits
