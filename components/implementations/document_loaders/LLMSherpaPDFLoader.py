from components.interfaces.all_interfaces import DocumentLoader

from pathlib import Path
from typing import List
import logging
import json
import os
from llmsherpa.readers import LayoutPDFReader
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from llmsherpa.readers import Document as Sherpa_Document
from langchain_core.documents import Document as Langchain_Document

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

        self.debug = False

    def _unstruct_partition_single_html(self, doc: Sherpa_Document, pdf_stem: str) -> List[Langchain_Document]:
        """Ingesting a Sherpa_Document, converting to HTML and parsing as HTML to a Langchain doc
        """

        html_doc = doc.to_html()

        try:
            elements = partition_html(text=html_doc)
            # v2 only works if there is a body and div doc etc.. "No <body class='Document'> or <div class='Page'> element found in the HTML.""
            # v2_elements = partition_html(
            #     text=html_doc, html_parser_version="v2", unique_element_ids=True
            # )
            if self.debug:
                for i, e in enumerate(elements):
                    if "This is where services like" in e.text:
                        print(i, "got ya")
        
        except Exception as e:
            self.logger.warning(f"Error in unstructured partition_html: {e}")

        chunks = chunk_by_title(
            elements, 
            combine_text_under_n_chars=200,
            include_orig_elements=True, # used for metadata gathering
            max_characters = 150000
            )
        
        if self.debug:
            for chunk in chunks:
                if "This is where services like" in chunk.text:
                    print(i, "got ya")
        
        self.logger.debug(f"created {len(chunks)} chunks from {pdf_stem}", "cyan")

        # this is what we return
        result_document_list = []

        # metadata
        # need to implement similar metadata gathering as in markdown, i.e. book title, last header / subtitle etc..
        html_meta = {"source": pdf_stem}

        
        for chunk in chunks:
            result_document_list.append(Langchain_Document(
                        page_content=str(chunk),
                        metadata=html_meta
                    ))

        return result_document_list
        
    def load(self) -> List[Langchain_Document]:
        """Reads all PDF files from the input dir, calls LLMSherpa PDF parser (_parse_pdf) which 
        outputs llmsherpa.readers.Document formats, then pass to unstructured to parse it as HTML
        """

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
                        lc_docs = self._unstruct_partition_single_html(doc, str(pdf.stem)) 
                        all_docs.extend(lc_docs)
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
                            self.logger.warning(f"{pdf_path} needs OCR, won't be parsed since OCR is disabled, moving on.")
                except Exception as e:
                    self.logger.error(f"Unhandled exception when reading single PDF file for {pdf}: {e}")

            if len(all_docs) > 0:
                self.logger.info(f"Successfully loaded {len(all_docs)} document(s) from {num_files} pdfs(s)")
            if self.debug:
                saveout = []
                for d in all_docs:
                    saveout.append({"content-length": len(d.page_content), "content": d.page_content})
                with open("chunks-before-splitting.json", "w", encoding="utf-8") as f:
                    json.dump(saveout, f, indent=2, ensure_ascii=False)
            return all_docs
    
        except Exception as e:
            self.logger.error(f"Error loading Markdown files from {self.file_path}: {e}")
            return []