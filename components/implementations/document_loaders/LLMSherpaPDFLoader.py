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


logger = logging.getLogger(__file__)


class PDFLoader(DocumentLoader):

    def __init__(self, file_path: str, api_url: str, api_url_ocr: str) -> None:
        self.file_path = Path(file_path)
        self.api_url = api_url
        self.api_url_ocr = api_url_ocr
        self.pdf_reader = LayoutPDFReader(self.api_url)
        self.do_ocr = False
        self.pdf_reader_ocr = None
        if self.do_ocr:
            self.pdf_reader_ocr = LayoutPDFReader(self.api_url_ocr)

        self.debug = False

    def conntest(self, filepath_or_url: str, max_retries: int = 4, sleep_time: int = 3) -> None:
        """Tries to parse a known good PDF URL (like Google's PDF) to verify if the NLM ingester is responsive."""
        retried = 0

        while retried < max_retries:
            try:
                result = self.pdf_reader.read_pdf(path_or_url=filepath_or_url)
                if isinstance(result.json[0], dict):
                    logger.info("Established connection with NLM-INGESTOR at endpoint %s", self.api_url)
                    return
            except LocationValueError:
                logger.error("Invalid API URL")
                sys.exit(f"Invalid API URL: {self.api_url}")
            except (NewConnectionError, MaxRetryError) as err:
                logger.error("NLM-INGESTOR connection test failed with an error", exc_info=err.args[0])

            retried += 1
            time.sleep(sleep_time)

        logger.error("NLM-INGESTOR connection test failed after retries. Exiting.")
        sys.exit(1)

    def _unstruct_partition_single_html(self, doc: Sherpa_Document, file_name: str) -> list[Langchain_Document]:
        """Ingesting a Sherpa_Document, converting to HTML and parsing as HTML to a Langchain doc"""
        try:
            elements = partition_html(text=doc.to_html())
            # v2 only works if there is a body and div doc etc.. "No <body class='Document'> or <div class='Page'> element found in the HTML.""
            # v2_elements = partition_html(
            #     text=html_doc, html_parser_version="v2", unique_element_ids=True
            # )
            if self.debug:
                for i, e in enumerate(elements):
                    if "This is where services like" in e.text:
                        print(i, "got ya")
        except Exception as e:
            logger.warning("Error in unstructured partition_html: %s", e)

        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=200,
            include_orig_elements=True,  # used for metadata gathering
            max_characters=150_000
        )

        if self.debug:
            for chunk in chunks:
                if "This is where services like" in chunk.text:
                    print(i, "got ya")

        logger.debug("created %s chunks from %s", len(chunks), file_name)

        # Create similar metadata as in case of markdown, i.e. book title, last header / subtitle etc..
        html_meta = {"source": file_name}

        return [
            Langchain_Document(page_content=str(chunk), metadata=html_meta)
            for chunk in chunks
        ]

    def load(self) -> list[Langchain_Document]:
        """
        Reads all PDF files from the input dir, calls LLMSherpa PDF parser (_parse_pdf) which
        outputs llmsherpa.readers.Document formats, then pass to unstructured to parse it as HTML
        """
        if not self.file_path.is_dir():
            logger.error("Provided path is not a directory: %s", self.file_path)
            return []

        all_docs = []

        for enum, pdf_path in enumerate(self.file_path.rglob("*.pdf"), start=1):
            with open(pdf_path, "rb") as f:
                file_data = f.read()

            pdf_file = (pdf_path.name, file_data, "application/pdf")
            try:
                parser_response = self.pdf_reader._parse_pdf(pdf_file)
                response_json = json.loads(parser_response.data.decode("utf-8"))
                blocks = response_json["return_dict"]["result"]["blocks"]
                # blocks are the extracted elements. If there are none, the reading has failed
                if blocks:
                    # encapsulate blocks to a llmsherpa.readers.Document object
                    doc = Sherpa_Document(blocks)
                    lc_docs = self._unstruct_partition_single_html(doc, pdf_path.name)
                    all_docs.extend(lc_docs)
                else:
                    if self.do_ocr:
                        logger.info("Could not parse %s, trying with OCR.", pdf_path)
                        parser_response = self.pdf_reader_ocr._parse_pdf(pdf_file)
                        response_json = json.loads(parser_response.data.decode("utf-8"))
                        blocks = response_json['return_dict']['result']['blocks']
                        if blocks:
                            all_docs.extend(Sherpa_Document(blocks))
                        else:
                            logger.warning("Could not parse %s, even with OCR, moving on...", pdf_path)
                    else:
                        logger.warning("%s needs OCR, won't be parsed since OCR is disabled, moving on.", pdf_path)
            except Exception as e:
                logger.error("Unhandled exception when reading PDF file %s: %s", pdf_path, e)

        logger.info("%s documents has been created from %s PDF", len(all_docs), enum)

        if self.debug:
            saveout = [
                {"content-length": len(d.page_content), "content": d.page_content}
                for d in all_docs
            ]
            with open("parsed-pdf-chunks.json", "w", encoding="utf-8") as f:
                json.dump(saveout, f, indent=2, ensure_ascii=False)

        return all_docs
