from dependency_injector import containers, providers
from components.implementations.document_loaders.markdown_loader import MarkdownLoader
from components.implementations.document_loaders.pdf_loader import PDFLoader
from components.implementations.document_splitters.splitter import MDSplitter
from components.implementations.embedders.openai_embedder import OpenAIEmbedder
from components.implementations.vectordb_inserters.chroma_inserter import ChromaDBRemoteInserter
from components.services.vector_indexer import VectorIndexer


class Container(containers.DeclarativeContainer):
    """Dependency Injection Container."""

    config = providers.Configuration()

    # Document Loader Provider
    document_loader = providers.Selector(
        config.file_type,
        markdown=providers.Singleton(
            MarkdownLoader,
            file_path=config.documentloader.file_path,
            max_workers=config.documentloader.max_workers
        ),
        pdf=providers.Singleton(
            PDFLoader,
            file_path=config.documentloader.file_path  # Assuming similar updates
        )
    )
    
    # Document Splitter Provider with Config Injection
    document_splitter = providers.Singleton(
        MDSplitter,
        chunk_size=config.splitter.chunk_size,
        chunk_overlap=config.splitter.chunk_overlap,
        headers_to_split_on=config.splitter.headers_to_split_on
    )

    # Embedder Provider
    embedder = providers.Factory(
        OpenAIEmbedder,
        model=config.embedder.openai.model, 
        batch_size=config.embedder.openai.batch_size
    )

    # Vector Inserter Provider
    vector_inserter = providers.Singleton(
        ChromaDBRemoteInserter,
        url=config.chroma.url,
        port=config.chroma.port,
        tenant=config.chroma.tenant,
        database=config.chroma.database,
        collection=config.chroma.collection,
        embedding_model=config.embedder.openai.model
    )

    # Vector Indexer Service Provider
    vector_indexer_service = providers.Singleton(
        VectorIndexer,
        embedder=embedder,
        document_loader=document_loader,
        document_splitter=document_splitter,
        vector_inserter=vector_inserter
    )