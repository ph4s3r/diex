from dependency_injector import containers, providers

from components.implementations.document_loaders.markdown_loader import MarkdownLoader
from components.implementations.document_loaders.pdf_loader import PDFLoader
from components.implementations.document_splitters.splitter import MDSplitter
from components.implementations.embedders.openai_embedder import OpenAIEmbedder
from components.implementations.vectordb_inserters.chroma_inserter import ChromaDBRemoteInserter


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()   

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

    embedder = providers.Factory(
        OpenAIEmbedder,
        model=config.embedder.openai.model, 
        batch_size=config.embedder.openai.batch_size
    )
    
    vector_inserter = providers.Factory(
        ChromaDBRemoteInserter,
        url=config.chroma.url,
        port=config.chroma.port,
        tenant=config.chroma.tenant,
        database=config.chroma.database,
        collection=config.chroma.collection,
        embedding_model=config.embedder.openai.model
    )

    vector_indexer_service = providers.Factory(
        'components.services.vector_indexer.VectorIndexer',
        document_loader=document_loader,
        document_splitter=document_splitter,
        embedder=embedder,
        vector_inserter=vector_inserter
    )