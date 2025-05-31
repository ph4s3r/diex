import logging
import requests
from pathlib import Path

from langchain_core.documents import Document

from components.interfaces.all_interfaces import DocumentLoader


class UnstructuredMDLoader(DocumentLoader):
    def __init__(
            self, 
            file_path: str, 
            url_source_stub: str = "",
            url_web_stub: str = "",
            url_check_all: str = "False"
            ) -> None:
        """
        Initializes the MarkdownLoader with a directory path for parallel loading.
        https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document
        https://python.langchain.com/docs/integrations/document_loaders/unstructured_markdown/
        https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html

        Args:
            file_path (str): The path to the directory containing Markdown files.
        """
        self.url_source_stub = url_source_stub
        self.url_web_stub = url_web_stub
        self.file_path: Path = Path(file_path).resolve()
        self.logger: logging.Logger = logging.getLogger("DocumentLoader")
        self.example_sources_shown = False
        self.url_check_all = url_check_all
        self.debug = False
        if self.debug:
            self.example_sources_shown = True
        self.meta_source_warned = False

   
    def fread(self, filename) -> str:
        with open(filename, encoding="utf8") as f:
            contents = f.read()
            if isinstance(contents, bytes):
                text = contents.decode("utf-8")
            else:
                text = contents
        return text
    
    def build_md_source_and_web_urls(self, md_file) -> dict:

        md_meta = {}  # all the custom metadata we gather manually from the docs
        # warning, this need to be checked every time
        try:
            md_meta["source_url"] = self.url_source_stub + str(md_file.relative_to(self.file_path)).replace("\\", "/")
            md_meta["web_url"] = self.url_web_stub + str(md_file.relative_to(self.file_path).with_suffix('')).replace("\\", "/")
            if "_index" in md_meta["web_url"]:
                md_meta["web_url"] = md_meta["web_url"].replace("_index", "")
        except Exception as e:
            self.meta_source_warned = True
            self.logger.warning(
                f"meta source cannot be compiled from the file ({str(md_file)}) because something is missing: {e}"
            )
        if not self.example_sources_shown:
            self.logger.info(f"sample source url: {md_meta['source_url']}")
            self.logger.info(f"sample web url: {md_meta['web_url']}")
            r = requests.get(md_meta["web_url"])
            if r.status_code == 404:
                self.logger.warning(f"Got 404 for web_url {md_meta["web_url"]} ")
            r = requests.get(md_meta["source_url"])
            if r.status_code == 404:
                self.logger.warning(f"Got 404 for source_url {md_meta["source_url"]} ")
            self.example_sources_shown = True

        if self.url_check_all:
            r = requests.get(md_meta["web_url"])
            if r.status_code == 404:
                self.logger.warning(f"Got 404 for web_url {md_meta["web_url"]} ")


        return md_meta

    def conntest(self) -> None:
        """Unused function stub for compatibility
        """
        return None

    def load(self) -> list[Document]:
        """
        Recursively loads all Markdown files from the directory and returns them as a list of Document objects.

        Returns:
            list[Document]: A list of loaded documents.
        """
        docs = []

        try:
            if not self.file_path.exists():
                self.logger.error(f"Directory does not exist: {self.file_path}")
                return []
            if not self.file_path.is_dir():
                self.logger.error(f"Provided path is not a directory: {self.file_path}")
                return []
        except Exception as e:
            self.logger.error(
                f"Unhandled exception loading directory {self.file_path}: {e}"
            )

        # Recursively find all .md files
        markdown_files = list(self.file_path.rglob("*.md"))
        num_files = len(markdown_files)
        self.logger.info(f"Found {num_files} Markdown file(s) in {self.file_path}")

        for mdfile in markdown_files:

            document = Document(
                page_content=self.fread(mdfile),
                metadata=self.build_md_source_and_web_urls(mdfile)
            )
            if len(document.page_content) == 0:
                self.logger.warning(f"{str(mdfile)}, has zero length, might worth checking out.")
            else:
                self.logger.debug(f"Read {str(mdfile)}, len: {len(document.page_content)} chars")
                docs.append(document)

        return docs

    
