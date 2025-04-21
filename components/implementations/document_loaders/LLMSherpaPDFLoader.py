import logging
from pathlib import Path
import time
import json
import sys

from urllib3.exceptions import NewConnectionError, MaxRetryError, LocationValueError
from llmsherpa.readers import LayoutPDFReader, Document as Sherpa_Document
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document as Langchain_Document

from components.interfaces.all_interfaces import DocumentLoader


class PDFLoader(DocumentLoader):

    def __init__(self, file_path: str, api_url: str, api_url_ocr: str) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.is_dir():
            sys.exit(f"The PDF source folder ({file_path}) is not a valid directory!")

        self.api_url = api_url
        self.api_url_ocr = api_url_ocr
        self.pdf_reader = LayoutPDFReader(self.api_url)
        self.pdf_reader_ocr = LayoutPDFReader(self.api_url_ocr)
        self.debug = False
        self.dummy_pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        self.logger: logging.Logger = logging.getLogger('DocumentLoader')

    def conntest(self, max_retries: int = 4, sleep_time: int = 3) -> None:
        """
        Tries to parse a known good PDF URL like Google's
        PDF to verify if the NLM ingester is responsive.
        """
        retried = 0

        while retried < max_retries:
            try:
                result = self.pdf_reader.read_pdf(self.dummy_pdf_url)
                if isinstance(result.json[0], dict):
                    self.logger.info("Established connection with NLM-INGESTOR at endpoint %s", self.api_url)
                    return
            except LocationValueError:
                self.logger.error("Invalid API URL")
                sys.exit(f"Invalid API URL: {self.api_url}")
            except (NewConnectionError, MaxRetryError) as err:
                self.logger.error("NLM-INGESTOR connection test failed with an error", exc_info=err.args[0])

            retried += 1
            time.sleep(sleep_time)

        self.logger.error("NLM-INGESTOR connection test failed after retries. Exiting.")
        sys.exit(1)

    def chunk_pdf_content(self, doc: Langchain_Document) -> list[Langchain_Document]:
        """Ingesting a Langchain_Document, converting to HTML and parsing as HTML to a Langchain doc"""
        try:
            # In case of an empty Sherpa blocks it generates an empty '<html></html>'
            elements = partition_html(text=doc.page_content)
            # v2 only works if there is a body and div doc etc.. "No <body class='Document'> or <div class='Page'> element found in the HTML.""
            # v2_elements = partition_html(text=html_doc, html_parser_version="v2", unique_element_ids=True)
            if self.debug:
                for i, e in enumerate(elements):
                    if "This is where services like" in e.text:
                        print(i, "got ya")
        except Exception:
            self.logger.exception("Error in unstructured partition_html %s", doc.metadata["source"])
            return []

        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=200,
            include_orig_elements=True,  # used for metadata gathering
            max_characters=150_000
        )
        self.logger.debug("created %s chunks from %s", len(chunks), doc.metadata["source"])

        if self.debug:
            for chunk in chunks:
                if "This is where services like" in chunk.text:
                    print(i, "got ya")

        return [
            Langchain_Document(page_content=str(chunk), metadata=doc.metadata)
            for chunk in chunks
        ]

    def process_pdf(self, file_path: Path) -> Sherpa_Document | None:
        """Read PDF file with Sherpa."""
        with open(file_path, "rb") as f:
            file_data = f.read()

        pdf_file = (file_path.name, file_data, "application/pdf")

        # No try-except block, haven't seen throwing exception even in case of a raster PDF
        # It returns with: {'return_dict': {'num_pages': 0, 'page_dim': [595.28, 841.89],
        #   'result': {'blocks': [], 'styles': []}}, 'status': 200}
        parser_response = self.pdf_reader._parse_pdf(pdf_file)
        response_json = json.loads(parser_response.data.decode("utf-8"))

        # blocks are the extracted elements. If there are none, the reading has failed
        if blocks := response_json["return_dict"]["result"]["blocks"]:
            return Sherpa_Document(blocks)

        self.logger.info("Could not parse %s, trying with OCR.", file_path.name)
        parser_response = self.pdf_reader_ocr._parse_pdf(pdf_file)
        response_json = json.loads(parser_response.data.decode("utf-8"))

        if blocks := response_json["return_dict"]["result"]["blocks"]:
            return Sherpa_Document(blocks)

        self.logger.warning("Could not parse even with OCR, moving on...")
        return None

    def sherpa2langchain(self, sdoc: Sherpa_Document, file_path: Path) -> Langchain_Document:

        return Langchain_Document(
                    page_content=sdoc.to_html(),
                    metadata={"source": file_path}
                )

    def load(self) -> list[Langchain_Document]:
        """
        Reads all PDF files from the input dir, calls LLMSherpa PDF parser (_parse_pdf) which
        outputs llmsherpa.readers.Document formats, then pass to unstructured to parse it as HTML
        """
        docs = []

        for pdf_path in self.file_path.rglob("*.pdf"):
            sherpa_doc = self.process_pdf(pdf_path)
            if sherpa_doc:
                lc_doc = self.sherpa2langchain(sherpa_doc, pdf_path.name)
                chunks = self.chunk_pdf_content(lc_doc)
                docs.extend(chunks)

        return docs
