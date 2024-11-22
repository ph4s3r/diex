from components.interfaces.all_interfaces import DocumentLoader

from typing import List
from langchain_core.documents import Document

class PDFLoader(DocumentLoader):
    def load(self, path: str) -> List[Document]:
        print(f"Function is waiting for imeplementation :)")
        return None
        # return [Document(content="Markdown content")]