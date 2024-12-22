# Vector Indexer

## What it do

index documents to a vector database
![diex3](https://github.com/user-attachments/assets/4d3f581a-2f95-465e-b9f2-1d92e4a2d567)

**Types**

Document = langchain_core.documents.base.Document
Vector = [List[float]]

input: a path to files
output: result of indexing

## Current working implementations

**DocumentLoader:**

- MarkdownLoader with **langchain_community.document_loaders**: UnstructuredMarkdownLoader
**DocumentSplitter:**
- MDSplitter with **langchain.text_splitter**: RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
**Embedder:**
- OpenAIEmbedder with **langchain_openai** : OpenAIEmbeddings
- SentenceTransformerEmbedder with **sentence_transformers**: SentenceTransformer (the stella model that is getting downloaded is only 1024 dimensions)
- GeneralEmbeddingAPIClient: primarily made for the huggingface server and 'quest' server (!)
**VectorInserter:**
- ChromaDBRemoteInserter with **chromadb**
- PineConeUpserter with **pinecone**.grpc

## How it do

the vector indexer service defines an implementation-independent flow wiring together the component interfaces
