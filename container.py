from dependency_injector import containers, providers
from components.implementations.document_loaders.UnstructuredMDLoader import UnstructuredMDLoader
from components.implementations.document_loaders.LLMSherpaPDFLoader import PDFLoader
from components.implementations.document_splitters.TiktokenRecursiveSplitter import TiktokenRecursiveSplitter
from components.implementations.embedders.voyage import VoyageEmbeddingAPIClient
from components.implementations.vectordb_inserters.pinecone_upserter import PineConeUpserter
from components.services.vector_indexer import VectorIndexer


class Container(containers.DeclarativeContainer):
    """Dependency Injection Container."""

    config = providers.Configuration()

    document_loader = providers.Selector(
        config.file_type,
        markdown=providers.Singleton(
            UnstructuredMDLoader,
            file_path=config.documentloader.file_path,
            max_workers=config.documentloader.max_workers,
            project=config.documentloader.project,
            version=config.documentloader.version
        ),
        pdf=providers.Singleton(
            PDFLoader,
            file_path=config.documentloader.file_path,
            api_url=config.documentloader.api_url,
            api_url_ocr=config.documentloader.api_url_ocr
        )
    )
    
    document_splitter = providers.Singleton(
        TiktokenRecursiveSplitter,
        max_token_seq_len=config.splitter.tokensplitter.max_token_seq_len,
        token_overlap=config.splitter.tokensplitter.token_overlap
    )

    embedder = providers.Singleton(
        VoyageEmbeddingAPIClient,
        output_dimension=config.embedder.voyage.output_dimension,
        output_dtype=config.embedder.voyage.output_dtype,
        model=config.embedder.voyage.model,
        batch_size=config.embedder.voyage.batch_size
    )

    vector_inserter = providers.Singleton(
        PineConeUpserter,
        api_key=config.pinecone.api_key,
        index_name=config.pinecone.index_name,
        index_host_suffix=config.pinecone.index_host_suffix,
        namespace=config.pinecone.namespace,
    )

    vector_indexer_service = providers.Singleton(
        VectorIndexer,
        embedder=embedder,
        document_loader=document_loader,
        document_splitter=document_splitter,
        vector_inserter=vector_inserter
    )