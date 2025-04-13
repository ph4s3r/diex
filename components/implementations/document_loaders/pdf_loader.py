from components.interfaces.all_interfaces import DocumentLoader

from pathlib import Path
from typing import List
import logging
import json
import os
from langchain_core.documents import Document as Langchain_Document
from llmsherpa.readers import LayoutPDFReader
from llmsherpa.readers import Document as Sherpa_Document

class PDFLoader(DocumentLoader):

    def __init__(
            self, 
            file_path: str,
            api_url: str,
            api_url_ocr: str,
            ) -> None:

        self.file_path: Path = Path(file_path).resolve()
        self.logger: logging.Logger = logging.getLogger('DocumentLoader')
        self.api_url = api_url
        self.api_url_ocr = api_url_ocr
        self.pdf_reader = LayoutPDFReader(self.api_url)
        self.do_ocr = False
        self.pdf_reader_ocr = None
        if self.do_ocr:
            self.pdf_reader_ocr = LayoutPDFReader(self.api_url_ocr)
        
    def load(self) -> List[Langchain_Document]:

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

            all_docs: List[Langchain_Document] = []

            for pdf in pdfs:
                try: 
                    pdf_path = str(pdf)
                    file_name = os.path.basename(pdf_path)
                    with open(pdf_path, "rb") as f:
                        file_data = f.read()
                        pdf_file = (file_name, file_data, 'application/pdf')
                    parser_response = self.pdf_reader._parse_pdf(pdf_file)
                    response_json = json.loads(parser_response.data.decode("utf-8"))
                    blocks = response_json['return_dict']['result']['blocks']
                    # blocks are the extracted elements. If there are none, the reading has failed
                    if len(blocks) > 0:
                        # encapsulate blocks to a llmsherpa.readers.Document object
                        doc = Sherpa_Document(blocks)
                        doc.sections()[0].to_html(include_children=True, recurse=True)
                        doc.sections()[1].block_json
                        doc.sections()[0].to_text()
                        doc.sections()[1].bbox
                        # llmsherpa.readers.Layout
                        html_doc = doc.to_html()
                        
                        # maybe we need here to parse HTML...

                        all_docs.extend(Sherpa_Document(blocks))
                    else:
                        if self.do_ocr:
                            self.logger.info(f"Could not parse {pdf_path}, trying with OCR.")
                            parser_response = self.pdf_reader_ocr._parse_pdf(pdf_file)
                            response_json = json.loads(parser_response.data.decode("utf-8"))
                            blocks = response_json['return_dict']['result']['blocks']
                            if len(blocks) > 0:
                                all_docs.extend(Sherpa_Document(blocks))
                            else:
                                self.logger.warning(f"Could not parse {pdf_path}, even with OCR, moving on..")
                        else:
                            self.logger.warning(f"Could not parse {pdf_path}; OCR is disabled, moving on.")
                except Exception as e:
                    self.logger.error(f"Unhandled exception when reading single PDF file for {pdf}: {e}")

            if len(all_docs) > 0:
                self.logger.info(f"Successfully loaded {len(all_docs)} document(s) from {num_files} pdfs(s)")
            return all_docs
    
        except Exception as e:
            self.logger.error(f"Error loading Markdown files from {self.file_path}: {e}")
            return []