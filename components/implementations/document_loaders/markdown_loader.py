from components.interfaces.all_interfaces import DocumentLoader

import logging
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader


class MarkdownLoader(DocumentLoader):

    def __init__(self):
            self.logger = logging.getLogger('DocumentLoader')

    def load(self, file_path: Path) -> List[Document]:
        """Loads Markdown files from the path and returns them as a list of langchain_core.documents.base.Document type
        https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document
        https://python.langchain.com/docs/integrations/document_loaders/unstructured_markdown/
        https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html
        """
        try:
            loader = UnstructuredMarkdownLoader(
                file_path,
                mode="single"  # one doc from one markdown
            )
            docs = loader.load()

            if not docs:
                self.logger.warning(f"Warning: No content loaded from {str(file_path)}")
                return []

            for doc in docs:
                if doc:
                    # Retain source filename as metadata
                    doc.metadata = {"source": str(file_path)}
                    self.logger.info(f"markdown file indexed: {str(file_path)}")

            return docs

        except Exception as e:
            self.logger.error(f"Error loading {str(file_path)}: {e}")
            return []