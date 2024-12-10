from components.interfaces.all_interfaces import DocumentLoader

import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from unstructured.partition.md import partition_md


class UnstructuredMDLoader(DocumentLoader):
    def __init__(
            self, 
            file_path: str, 
            max_workers: Optional[int] = None, 
            project: str = None
            ) -> None:
        """
        Initializes the MarkdownLoader with a directory path and optional max_workers for parallel loading.
        https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document
        https://python.langchain.com/docs/integrations/document_loaders/unstructured_markdown/
        https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html

        Args:
            file_path (str): The path to the directory containing Markdown files.
            max_workers (Optional[int]): The maximum number of threads to use for parallel loading.
                                        Defaults to the number of processors on the machine, multiplied by 5.
        """
        self.file_path: Path = Path(file_path).resolve()
        self.logger: logging.Logger = logging.getLogger('DocumentLoader')

    def load(self) -> List[Document]:
        """
        Recursively loads all Markdown files from the directory and returns them as a list of Document objects.

        Returns:
            List[Document]: A list of loaded documents.
        """
        docs = None

        try:
            if not self.file_path.exists():
                self.logger.error(f"Directory does not exist: {self.file_path}")
                return []
            if not self.file_path.is_dir():
                self.logger.error(f"Provided path is not a directory: {self.file_path}")
                return []
        except:
            self.logger.error(f"Unhandled exception loading directory {self.file_path}: {e}")

        # Recursively find all .md files
        markdown_files = list(self.file_path.rglob('*.md'))
        num_files = len(markdown_files)
        self.logger.info(f"Found {num_files} Markdown file(s) in {self.file_path}")
        

        for mdfile in markdown_files:
            mdelements = self._unstruct_partition_single_md(mdfile)
            docs.append(mdelements) # lehet extend kell
        return docs
    

    def _unstruct_partition_single_md(self, md_file: Path) -> Optional[List[Document]]:

        elements = partition_md(filename=md_file)

        return elements
