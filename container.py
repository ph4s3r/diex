import yaml
from dependency_injector import containers, providers

from components.implementations.document_loaders.markdown_loader import MarkdownLoader
from components.implementations.document_loaders.pdf_loader import PDFLoader
from components.implementations.document_splitters.splitter import MDSplitter

class Container(containers.DeclarativeContainer):
    
    config = providers.Configuration(yaml_files=["configs/config.yaml"])

    def load_config(self, config_path: str):
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        self.config.from_dict(config_data)
    
    document_loader = providers.Selector(
        config.file_type,
        markdown=providers.Factory(MarkdownLoader),
        pdf=providers.Factory(PDFLoader)
    )
    
    document_splitter = providers.Factory(
        MDSplitter,
        chunk_size=config.splitter.chunk_size,
        chunk_overlap=config.splitter.chunk_overlap,
        headers_to_split_on=config.splitter.headers_to_split_on
    )
    
    vector_indexer_service = providers.Factory(
        'components.services.vector_indexer.VectorIndexer',
        document_loader=document_loader,
        document_splitter=document_splitter
    )