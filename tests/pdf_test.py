"""Test parsing PDF files."""
from pathlib import Path

import pytest

from components.implementations.document_loaders.LLMSherpaPDFLoader import PDFLoader

PDF_TEST_FOLDER = Path(__file__).parent / "test_files"
API_URL = "http://localhost:5010/api/parseDocument?renderFormat=all"


def test_conntest_with_invalid_api_url():
    """Test connection checker method."""
    api_url = "/api/parseDocument?renderFormat=all"
    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    loader = PDFLoader(file_path=PDF_TEST_FOLDER, api_url=api_url, api_url_ocr=None)

    with pytest.raises(SystemExit):
        loader.conntest(filepath_or_url=test_url, sleep_time=1)


def test_conntest_with_incorrect_api_url():
    """Test connection checker method."""
    api_url = "http://localhost:9010/api/parseDocument?renderFormat=all"
    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    loader = PDFLoader(file_path=PDF_TEST_FOLDER, api_url=api_url, api_url_ocr=None)

    with pytest.raises(SystemExit):
        loader.conntest(filepath_or_url=test_url, sleep_time=1)


def test_conntest():
    """Test connection checker method."""
    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    loader = PDFLoader(file_path=PDF_TEST_FOLDER, api_url=API_URL, api_url_ocr=None)
    assert loader.conntest(filepath_or_url=test_url) is None


def test_with_vector_pdf():
    """Test parsing a single page vector PDF."""
    loader = PDFLoader(file_path=PDF_TEST_FOLDER, api_url=API_URL, api_url_ocr=None)
    docs = loader.load()
    assert docs
