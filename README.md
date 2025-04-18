# Vector Indexer

## What it do

index various documents to a vector database
![diex3](https://github.com/user-attachments/assets/4d3f581a-2f95-465e-b9f2-1d92e4a2d567)

## Current implementation

**DocumentLoader:**

- Markdown Loader using UnstructuredMarkdownLoader
- PDF Loader using LLMSherpa locally (ghcr.io/nlmatics/nlm-ingestor:latest) converting PDF to HTML, then parsing HTML with Unstructured

**DocumentSplitter:**
- TikToken based recursive splitter

**Embedder:**
- VoyageAI Embedder

**VectorInserter:**
- PineConeInserter with **pinecone**.grpc

## How it do

the vector indexer service (services/vector_indexer.py) defines an implementation-independent flow wiring together the component interfaces and running them sequentially
