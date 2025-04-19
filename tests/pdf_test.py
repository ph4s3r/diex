"""Test parsing PDF files."""
from pathlib import Path

import pytest
from llmsherpa.readers import Document

from components.implementations.document_loaders.LLMSherpaPDFLoader import PDFLoader

PDF_TEST_FOLDER = Path(__file__).parent / "test_files"
API_URL = "http://localhost:5010/api/parseDocument?renderFormat=all"


def test_conntest_with_invalid_api_url():
    """Test connection checker method."""
    api_url = "/api/parseDocument?renderFormat=all"

    loader = PDFLoader(file_path=PDF_TEST_FOLDER, api_url=api_url, api_url_ocr=None)

    with pytest.raises(SystemExit):
        loader.conntest(sleep_time=1)


def test_conntest_with_incorrect_api_url():
    """Test connection checker method."""
    api_url = "http://localhost:9010/api/parseDocument?renderFormat=all"

    loader = PDFLoader(file_path=PDF_TEST_FOLDER, api_url=api_url, api_url_ocr=None)

    with pytest.raises(SystemExit):
        loader.conntest(sleep_time=1)


def test_conntest():
    """Test connection checker method."""

    loader = PDFLoader(file_path=PDF_TEST_FOLDER, api_url=API_URL, api_url_ocr=None)
    assert loader.conntest() is None


def test_with_vector_pdf():
    """Test parsing a single page vector PDF."""
    test_pdf = PDF_TEST_FOLDER / "LHH_CH_SalaryGuide_2024.pdf"
    assert test_pdf.exists()

    loader = PDFLoader(file_path=PDF_TEST_FOLDER, api_url=API_URL, api_url_ocr=None)
    sherpa_doc = loader.process_pdf(test_pdf)
    assert isinstance(sherpa_doc, Document)
    assert sherpa_doc.to_text(), "Error parsing PDF, content is expected"
    # Do other content related tests if it would have benefit


def test_with_raster_pdf():
    """Test parsing a single page raster PDF."""
    test_pdf = PDF_TEST_FOLDER / "LHH_CH_SalaryGuide_2024_raster.pdf"
    assert test_pdf.exists()
    api_url_ocr = "http://localhost:5010/api/parseDocument?renderFormat=all&applyOcr=yes"

    loader = PDFLoader(file_path=PDF_TEST_FOLDER, api_url=API_URL, api_url_ocr=api_url_ocr)
    sherpa_doc = loader.process_pdf(test_pdf)
    assert isinstance(sherpa_doc, Document)
    # FYI: it does not work as expected, the implementation need to be fixed
    assert sherpa_doc.to_text(), "Error parsing PDF, content is expected"
