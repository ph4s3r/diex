import logging
from components.interfaces.all_interfaces import DocumentLoader
from components.interfaces.all_interfaces import DocumentSplitter

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class VectorIndexer:
    def __init__(
        self,
        document_loader: DocumentLoader,
        document_splitter: DocumentSplitter
    ):
        self.document_loader = document_loader
        self.document_splitter = document_splitter
        self.logger = logging.getLogger('VectorIndexer')

    def process(self, path: str) -> None:


        self.logger.info("BEGIN LOADING")
        documents = self.document_loader.load(path)
        number_of_docs_indexed = len(documents)
        self.logger.info(f"LOADED {number_of_docs_indexed} file from {path}")
        assert number_of_docs_indexed is not None and number_of_docs_indexed != 0

        self.logger.info("BEGIN SPLIT")
        split_documents = self.document_splitter.split(documents)
        number_of_docs_split = len(split_documents)
        self.logger.info(f"RESULTS OF THE SPLIT IS {number_of_docs_split} DOCUMENTS")
        assert number_of_docs_split is not None and number_of_docs_split != 0
        
        self.logger.info("END")
