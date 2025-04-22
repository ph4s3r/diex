from dependency_injector import containers, providers

from components.services.vector_indexer import VectorIndexer
from components.implementations.document_loaders.UnstructuredMDLoader import UnstructuredMDLoader
from components.implementations.document_loaders.LLMSherpaPDFLoader import PDFLoader
from components.implementations.semantic_chunkers.UnstructuredChunkers import UnstructuredHTMLChunker, UnstructuredMarkdownChunker
from components.implementations.document_splitters.TiktokenRecursiveSplitter import TiktokenRecursiveSplitter
from components.implementations.embedders.VoyageEmbedder import VoyageEmbedder
from components.implementations.vectordb_inserters.PineConeInserter import PineConeInserter


class Container(containers.DeclarativeContainer):
    """Dependency Injection Container."""

    config = providers.Configuration()

    document_loader = providers.Selector(
        config.file_type,
        markdown=providers.Singleton(
            UnstructuredMDLoader,
            file_path=config.documentloader.file_path,
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

    document_chunker = providers.Selector(
        config.file_type,
        pdf=providers.Singleton(
            UnstructuredHTMLChunker
        ),
        markdown=providers.Singleton(
            UnstructuredMarkdownChunker,
            project=config.documentloader.project,
            version=config.documentloader.version
        )
    )    
    
    document_splitter = providers.Singleton(
        TiktokenRecursiveSplitter,
        max_token_seq_len=config.splitter.tokensplitter.max_token_seq_len,
        token_overlap=config.splitter.tokensplitter.token_overlap
    )

    embedder = providers.Singleton(
        VoyageEmbedder,
        output_dimension=config.embedder.voyage.output_dimension,
        output_dtype=config.embedder.voyage.output_dtype,
        model=config.embedder.voyage.model,
        batch_size=config.embedder.voyage.batch_size
    )

    vector_inserter = providers.Singleton(
        PineConeInserter,
        api_key=config.pinecone.api_key,
        index_name=config.pinecone.index_name,
        index_host_suffix=config.pinecone.index_host_suffix,
        namespace=config.pinecone.namespace,
    )

    vector_indexer_service = providers.Singleton(
        VectorIndexer,
        embedder=embedder,
        document_loader=document_loader,
        document_chunker=document_chunker,
        document_splitter=document_splitter,
        vector_inserter=vector_inserter
    )