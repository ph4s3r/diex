import logging
import json

from langchain_core.documents import Document
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

from components.interfaces.all_interfaces import SemanticChunker


class UnstructuredHTMLChunker(SemanticChunker):

    def __init__(self) -> None:
        
        self.debug = False
        self.logger: logging.Logger = logging.getLogger('SemanticChunker')


    def chunk(self, docs: list[Document]) -> list[Document]:
        """Ingesting a list of Langchain_Document types, and doing semantic chunking with unstructured
        """
        return_docs: list[Document] = []

        for doc in docs:

            try:
                # In case of an empty Sherpa blocks it generates an empty '<html></html>'
                elements = partition_html(text=doc.page_content)
                # v2 only works if there is a body and div doc etc.. "No <body class='Document'> or <div class='Page'> element found in the HTML.""
                # v2_elements = partition_html(text=html_doc, html_parser_version="v2", unique_element_ids=True)
                if self.debug:
                    for i, e in enumerate(elements):
                        if "This is where services like" in e.text:
                            print(i, "got ya")
            except Exception:
                self.logger.exception("Error in unstructured partition_html %s", doc.metadata["source"])
                return []

            chunks = chunk_by_title(
                elements,
                combine_text_under_n_chars=200,
                include_orig_elements=False,  # used for debugging & metadata gathering
                max_characters=150_000
            )
            self.logger.debug("created %s chunks from %s", len(chunks), doc.metadata["source"])

            if self.debug:
                for chunk in chunks:
                    if "This is where services like" in chunk.text:
                        print(i, "got ya")

            if self.debug:
                saveout = [
                    {"content-length": len(d.page_content), "content": d.page_content}
                    for d in docs
                ]
                with open("parsed-pdf-chunks.json", "w", encoding="utf-8") as f:
                    json.dump(saveout, f, indent=2, ensure_ascii=False)

            chunk_doc_list = [
                Document(page_content=str(chunk), metadata=doc.metadata)
                for chunk in chunks
            ]

            return_docs.extend(chunk_doc_list)
        
        return return_docs
    

class UnstructuredMarkdownChunker(SemanticChunker):

    def __init__(self) -> None:
        
        self.debug = False
        self.logger: logging.Logger = logging.getLogger('SemanticChunker')


    def chunk(self, documents: list[Document]) -> list[Document]:

        docs = documents

        if self.debug:
            saveout = [
                {"content-length": len(d.page_content), "content": d.page_content}
                for d in docs
            ]
            with open("parsed-md-chunks.json", "w", encoding="utf-8") as f:
                json.dump(saveout, f, indent=2, ensure_ascii=False)

        return documents