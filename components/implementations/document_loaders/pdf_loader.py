from components.interfaces.all_interfaces import DocumentLoader

from pathlib import Path
from typing import List
import logging
import json
import os
import pprint
from termcolor import cprint
from llmsherpa.readers import LayoutPDFReader
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from llmsherpa.readers import Document as Sherpa_Document
from langchain_core.documents import Document as Langchain_Document
from unstructured.staging.base import elements_from_base64_gzipped_json


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

    def _unstruct_partition_single_html(self, doc: Sherpa_Document, pdf_path: str) -> List[Langchain_Document]:
        """Ingesting a Sherpa_Document, converting to HTML and parsing as HTML to a Langchain doc"""


        # llmsherpa.readers.Layout
        doc.sections()[0].to_html(include_children=True, recurse=True)
        doc.sections()[1].block_json
        doc.sections()[0].to_text()
        doc.sections()[1].bbox
        html_doc = doc.to_html()
        try:
            elements = partition_html(text=html_doc)
        except Exception as e:
            self.logger.warning(f"Error in unstructured partition_html: {e}")

        chunks = chunk_by_title(
            elements, 
            combine_text_under_n_chars=0,
            include_orig_elements=True, # stores the elements under ["orig_elements"]
            max_characters=5000
            )
        
        self.logger.debug(f"created {len(chunks)} chunks from {pdf_path}", "cyan")

        # this is what we return
        result_document_list = []

        html_meta = {} # all the custom metadata we gather manually from the docs

        for chunk in chunks:
            md_h_list = [""] * 6 # markdown headers have a max depth of 6
            # TODO: get HTML header
            # md_h_list[0] = html_meta.get("main_header", "")
            chunk_meta = html_meta
            chunk_inmeta = chunk.metadata.to_dict()
            orig_elements = elements_from_base64_gzipped_json(chunk_inmeta["orig_elements"])
            # standalone header 
            # len(orig_elements[0].text) > 5 BUG!!! :
            # unstructured sometimes classifies a few standalone words or just a random line as a title, 
            # while our mechanism here relies on the promised title based splitting...
            if len(orig_elements) == 1 and orig_elements[0].category == 'Title' and len(orig_elements[0].text) > 5:
                # will jnot be saved as a separate chunk / vector - we just save it as a header (metadata)
                depth = orig_elements[0].metadata.category_depth
                # resetting the md headers until the current title depth (the depth 2 header resets an earlier depth 3 and below)
                md_h_list[depth:] = [""] * (len(md_h_list) - depth)
                md_h_list[orig_elements[0].metadata.category_depth] = orig_elements[0].text
                self.logger.debug(f"Standalone chunk: {orig_elements[0].text}")
            else:
                # all the markdown headers have a category depth = no. of hashtags - 1 - load them into the chunk meta
                for elem in orig_elements:
                    if elem.category == 'Title':
                        depth = elem.metadata.category_depth
                        md_h_list[depth:] = [""] * (len(md_h_list) - depth)
                        md_h_list[elem.metadata.category_depth] = elem.text
                # remove empty headers
                while len(md_h_list) > 0:
                    if md_h_list[-1] == "":
                        md_h_list.pop()
                    else:
                        break
                chunk_meta.pop('author', None)
                chunk_meta.pop('ms.author', None)
                if len(md_h_list) > 0:
                    markdown_header_struct_dict = {f"header_{i}": value for i, value in enumerate(md_h_list)}
                    chunk_meta.update(markdown_header_struct_dict)
                result_document_list.append(Langchain_Document(
                    page_content=str(chunk),
                    metadata=chunk_meta
                ))
                if self.debug:
                    cprint("  CHUNK META:", "red")
                    pprint.pprint(chunk_meta)
                    cprint("  DOC page_content:", "red")
                    cprint(str(chunk), "green")
                    cprint("\n\n" + "-"*80, "red")

        return result_document_list
        
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
                        lc_docs = self._unstruct_partition_single_html(doc, pdf_path) 
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
                            self.logger.warning(f"Could not parse {pdf_path}; OCR is disabled, moving on.")
                except Exception as e:
                    self.logger.error(f"Unhandled exception when reading single PDF file for {pdf}: {e}")

            if len(all_docs) > 0:
                self.logger.info(f"Successfully loaded {len(all_docs)} document(s) from {num_files} pdfs(s)")
            return all_docs
    
        except Exception as e:
            self.logger.error(f"Error loading Markdown files from {self.file_path}: {e}")
            return []