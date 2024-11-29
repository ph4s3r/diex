from dependency_injector import containers, providers
from components.implementations.document_loaders.markdown_loader import MarkdownLoader
from components.implementations.document_loaders.pdf_loader import PDFLoader
from components.implementations.document_splitters.splitter import MDSplitter
from components.implementations.embedders.openai_embedder import OpenAIEmbedder
from components.implementations.embedders.st_embedder import SentenceTransformerEmbedder
from components.implementations.vectordb_inserters.pinecone_upserter import PineConeUpserter
from components.services.vector_indexer import VectorIndexer


class Container(containers.DeclarativeContainer):
    """Dependency Injection Container."""

    config = providers.Configuration()

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
    
    document_splitter = providers.Singleton(
        MDSplitter,
        chunk_size=config.splitter.chunk_size,
        chunk_overlap=config.splitter.chunk_overlap,
        headers_to_split_on=config.splitter.headers_to_split_on
    )

    embedder = providers.Singleton(
        SentenceTransformerEmbedder,
        embedding_model=config.embedder.embedding_model
    )

    # embedder = providers.Singleton(
    #     OpenAIEmbedder,
    #     model=config.embedder.embedding_model,
    #     batch_size=config.embedder.openai.batch_size,
    #     max_tokens_per_minute=config.embedder.openai.max_tokens_per_minute,
    #     max_requests_per_minute=config.embedder.openai.max_requests_per_minute,
    #     batch_queue_limit=config.embedder.openai.batch_queue_limit
    # )

    vector_inserter = providers.Singleton(
        PineConeUpserter
    )

    vector_indexer_service = providers.Singleton(
        VectorIndexer,
        embedder=embedder,
        document_loader=document_loader,
        document_splitter=document_splitter,
        vector_inserter=vector_inserter
    )