from components.interfaces.all_interfaces import DocumentLoader

import logging
from typing import List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import os  # for cpu_count()


class MarkdownLoader(DocumentLoader):
    def __init__(self, file_path: str, max_workers: Optional[int] = None) -> None:
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
        self.max_workers: int = max_workers or (os.cpu_count() * 5 if os.cpu_count() else 10)

    def _load_single_file(self, md_file: Path) -> Optional[List[Document]]:
        """
        Loads a single Markdown file and returns the list of Document objects.

        Args:
            md_file (Path): The path to the Markdown file.

        Returns:
            Optional[List[Document]]: A list of Document objects if successful, else None.
        """
        try:
            self.logger.debug(f"Loading file: {md_file}")
            loader = UnstructuredMarkdownLoader(
                str(md_file),
                mode="single"  # one doc per markdown file
            )
            docs: List[Document] = loader.load()

            if not docs:
                self.logger.warning(f"No content loaded from {md_file}")
                return None

            for doc in docs:
                if doc:
                    # Retain source filename as metadata
                    doc.metadata = {"source": str(md_file)}
                    self.logger.debug(f"Indexed markdown file: {md_file}")

            # self.logger.info(f"Successfully loaded {len(docs)} document(s) from {md_file}")
            return docs

        except Exception as e:
            self.logger.error(f"Error loading {md_file}: {e}")
            return None

    def load(self) -> List[Document]:
        """
        Recursively loads all Markdown files from the directory and returns them as a list of Document objects.

        Returns:
            List[Document]: A list of loaded documents.
        """
        try:
            if not self.file_path.exists():
                self.logger.error(f"Directory does not exist: {self.file_path}")
                return []
            if not self.file_path.is_dir():
                self.logger.error(f"Provided path is not a directory: {self.file_path}")
                return []

            # Recursively find all .md files
            markdown_files = list(self.file_path.rglob('*.md'))
            num_files = len(markdown_files)
            self.logger.info(f"Found {num_files} Markdown file(s) in {self.file_path}")

            if num_files == 0:
                self.logger.warning(f"No Markdown files found in directory: {self.file_path}")
                return []

            all_docs: List[Document] = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all file loading tasks
                future_to_file = {executor.submit(self._load_single_file, md_file): md_file for md_file in markdown_files}

                for future in as_completed(future_to_file):
                    md_file = future_to_file[future]
                    try:
                        docs = future.result()
                        if docs:
                            all_docs.extend(docs)
                    except Exception as e:
                        self.logger.error(f"Unhandled exception for {md_file}: {e}")

            self.logger.info(f"Successfully loaded {len(all_docs)} document(s) from {num_files} file(s)")
            return all_docs

        except Exception as e:
            self.logger.error(f"Error loading Markdown files from {self.file_path}: {e}")
            return []
