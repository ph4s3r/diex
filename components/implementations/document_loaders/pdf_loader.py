from components.interfaces.all_interfaces import DocumentLoader

from pathlib import Path
from typing import List
import logging
import json
import os
from langchain_core.documents import Document
from llmsherpa.readers import LayoutPDFReader

class PDFLoader(DocumentLoader):

    def __init__(
            self, 
            file_path: str,
            api_url: str,
            ) -> None:

        self.file_path: Path = Path(file_path).resolve()
        self.logger: logging.Logger = logging.getLogger('DocumentLoader')
        self.api_url = api_url
        self.pdf_reader = LayoutPDFReader(self.api_url)
        
    def load(self) -> List[Document]:

        try:
            if not self.file_path.exists():
                self.logger.error(f"Directory does not exist: {self.file_path}")
                return []
            if not self.file_path.is_dir():
                self.logger.error(f"Provided path is not a directory: {self.file_path}")
                return []

            pdfs = list(self.file_path.rglob('*.pdf'))
            num_files = len(pdfs)
            if num_files == 0:
                self.logger.warning(f"No Markdown files found in directory: {self.file_path}")
                return []
            
            self.logger.info(f"Found {num_files} pdf file(s) in {self.file_path}")

            all_docs: List[Document] = []

            for pdf in pdfs:
                try: 

                    # this returned with some bogus error message...
                    # doc = self.pdf_reader.read_pdf(
                    #     path_or_url=str(pdf), 
                    #     contents=None
                    #     )
                    # so here is read_pdf fun from lib

                    path_or_url = str(pdf)
                    file_name = os.path.basename(path_or_url)
                    with open(path_or_url, "rb") as f:
                        file_data = f.read()
                        pdf_file = (file_name, file_data, 'application/pdf')
                    parser_response = self.pdf_reader._parse_pdf(pdf_file)
                    response_json = json.loads(parser_response.data.decode("utf-8"))
                    blocks = response_json['return_dict']['result']['blocks']

                    all_docs.extend(Document(blocks))
                except Exception as e:
                    self.logger.error(f"Unhandled exception when reading single PDF file for {pdf}: {e}")

            if len(all_docs) > 0:
                self.logger.info(f"Successfully loaded {len(all_docs)} document(s) from {num_files} pdfs(s)")
            return all_docs
    
        except Exception as e:
            self.logger.error(f"Error loading Markdown files from {self.file_path}: {e}")
            return []