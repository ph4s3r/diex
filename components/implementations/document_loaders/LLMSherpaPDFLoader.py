import logging
from pathlib import Path
import time
import json
import sys

from urllib3.exceptions import NewConnectionError, MaxRetryError, LocationValueError
from llmsherpa.readers import LayoutPDFReader, Document as Sherpa_Document
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


    def process_pdf(self, file_path: Path) -> Sherpa_Document | None:
        """Read PDF file with Sherpa: Creates the llmsherpa.readers.Document format from the PDF, 
        converts it to HTML and encapsulates it into a LangChain Document format
        """
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
            return Langchain_Document(
                    page_content=Sherpa_Document(blocks).to_html(),
                    metadata={"source": file_path}
                )

        self.logger.info("Could not parse %s, trying with OCR.", file_path.name)
        parser_response = self.pdf_reader_ocr._parse_pdf(pdf_file)
        response_json = json.loads(parser_response.data.decode("utf-8"))

        if blocks := response_json["return_dict"]["result"]["blocks"]:
            return Langchain_Document(
                    page_content=Sherpa_Document(blocks).to_html(),
                    metadata={"source": file_path}
                )

        self.logger.warning("Could not parse even with OCR, moving on...")
        return None


    def load(self) -> list[Langchain_Document]:
        """
        Reads all PDF files from the input dir, calls LLMSherpa PDF parser (process_pdf) 
        and puts the returned Langchain_Documents into a list (one per PDF)
        """
        docs = []

        for pdf_path in self.file_path.rglob("*.pdf"):
            langchain_doc = self.process_pdf(pdf_path)
            docs.append(langchain_doc)

        return docs
