import re
import json
import logging

import pprint
from langchain_core.documents import Document
from unstructured.partition.md import partition_md
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_from_base64_gzipped_json

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
                continue

            chunks = chunk_by_title(
                elements,
                combine_text_under_n_chars=200,
                include_orig_elements=False,  # used for debugging & metadata gathering
                max_characters=150_000
            )
            self.logger.debug("created %s chunks from %s", len(chunks), doc.metadata["source"])

            if self.debug:
                for i, chunk in enumerate(chunks):
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
                Document(page_content=chunk.text, metadata=doc.metadata)
                for chunk in chunks
            ]

            return_docs.extend(chunk_doc_list)
        
        return return_docs
    

class UnstructuredMarkdownChunker(SemanticChunker):

    def __init__(
            self,
            project: str = "",
            version: str = ""
            ) -> None:

        self.project = project
        self.version = version
        self.debug = False
        self.logger: logging.Logger = logging.getLogger('SemanticChunker')


    def _text_to_kv(self, input_string):
        """
        Parses the input string into a dictionary of key-value pairs.

        Args:
            input_string (str): The string containing key-value pairs separated by colons.

        Returns:
            dict: A dictionary containing all extracted key-value pairs.
        """
        # Regular expression pattern to match key-value pairs
        pattern = r"(\b[\w\.]+):\s*([^:]+)(?=\s+\b[\w\.]+:|$)"

        # Find all matches in the input string
        matches = re.findall(pattern, input_string)

        # Construct the dictionary from the matches
        kv_pairs = {key.strip(): value.strip() for key, value in matches}

        return kv_pairs


    def chunk(self, documents: list[Document]) -> list[Document]:
        # this is what we return
        return_docs: list[Document] = []
        errored_docs = []

        for doc in documents:

            md_meta = doc.metadata.copy()

            try:
                elements = partition_md(text=doc.page_content)
            except AttributeError:
                # usually there is this e.g. skipping processing markdown file of
                # https://github.com/MicrosoftDocs/azure-docs/tree/main/articles/virtual-network/what-is-ip-address-168-63-129-16.md:
                # 'lxml.etree._ProcessingInstruction' object has no attribute 'is_phrasing'
                errored_docs.append(md_meta["source"])
                continue
            except Exception:
                src = md_meta["source"]
                self.logger.exception("Error processing md file %s", src)
                errored_docs.append(src)
                continue
            
            # all microsoft learn docs start with a special header, we try to process these 3 first elements automatically
            if (
                any(substring in self.project for substring in ["azure", "microsoft"])
                and "includes" not in md_meta["source"]
            ):
                try:
                    msheaders = self._text_to_kv(elements[0].text)
                    # TODO: why do we remove this?
                    elements.pop(0).text
                    md_meta.update(msheaders)
                except Exception as e:
                    src = md_meta["source"]
                    self.logger.warning(
                        f"Skipping processing Microsoft markdown header of {src}: {e}"
                    )
                if len(elements) > 0:
                    if "intent" in elements[0].text and elements[0].category == "Title":
                        try:
                            intent = elements[0].text.split(":")[1:][0]
                            md_meta["intent"] = intent
                            elements.pop(0)
                        # then try another way
                        except:  # noqa: E722
                            try:
                                md_meta["intent"] = elements[0].text
                                elements.pop(0)
                            except Exception as e:
                                self.logger.warning(
                                    f"Could not process intent, we skip then the header processing, {e}"
                                )

                    # no intent doc, lets try with processing the first header if there is no intent
                    if elements[0].category == "Title":
                        md_meta["main_header"] = elements.pop(0).text
                else:
                    src = md_meta["source"]
                    self.logger.warning(f"Not much stuff in here, skipping {src}")

            chunks = chunk_by_title(
                elements,
                combine_text_under_n_chars=0,
                include_orig_elements=True,  # stores the elements under ["orig_elements"]
                max_characters=5000,
            )
            src = md_meta["source"]
            self.logger.debug(f"created {len(chunks)} chunks from {src}")

            for chunk in chunks:
                md_h_list = [""] * 6  # markdown headers have a max depth of 6
                md_h_list[0] = md_meta.get("main_header", "")
                chunk_meta = md_meta.copy()
                chunk_inmeta = chunk.metadata.to_dict()
                orig_elements = elements_from_base64_gzipped_json(
                    chunk_inmeta["orig_elements"]
                )
                # standalone header
                # len(orig_elements[0].text) > 5 BUG!!! :
                # unstructured sometimes classifies a few standalone words or just a random line as a title, while our mechanism here relies on the promised title based splitting...
                if (
                    len(orig_elements) == 1
                    and orig_elements[0].category == "Title"
                    and len(orig_elements[0].text) > 5
                ):
                    # will jnot be saved as a separate chunk / vector - we just save it as a header (metadata)
                    depth = orig_elements[0].metadata.category_depth
                    # resetting the md headers until the current title depth (the depth 2 header resets an earlier depth 3 and below)
                    md_h_list[depth:] = [""] * (len(md_h_list) - depth)
                    md_h_list[orig_elements[0].metadata.category_depth] = orig_elements[
                        0
                    ].text
                    self.logger.debug(f"Standalone chunk: {orig_elements[0].text}")
                else:
                    # all the markdown headers have a category depth = no. of hashtags - 1 - load them into the chunk meta
                    for elem in orig_elements:
                        if elem.category == "Title":
                            depth = elem.metadata.category_depth
                            md_h_list[depth:] = [""] * (len(md_h_list) - depth)
                            md_h_list[elem.metadata.category_depth] = elem.text
                    # remove empty headers
                    while len(md_h_list) > 0:
                        if md_h_list[-1] == "":
                            md_h_list.pop()
                        else:
                            break
                    chunk_meta.pop("author", None)
                    chunk_meta.pop("ms.author", None)
                    if len(md_h_list) > 0:
                        markdown_header_struct_dict = {
                            f"header_{i}": value for i, value in enumerate(md_h_list)
                        }
                        chunk_meta.update(markdown_header_struct_dict)
                    return_docs.append(
                        Document(page_content=chunk.text, metadata=chunk_meta)
                    )
                    if self.debug:
                        try:
                            from termcolor import cprint
                            cprint("  CHUNK META:", "red")
                            pprint.pprint(chunk_meta)
                            cprint("  DOC page_content:", "red")
                            cprint(chunk.text, "green")
                            cprint("\n\n" + "-" * 80, "red")
                        except ImportError:
                            self.logger.warning("from termcolor import cprint cannot be imported, no chunk DEBUG info can be displayed")
                            pass

        if self.debug:
            saveout = [
                {"content-length": len(d.page_content), "content": d.page_content}
                for d in return_docs
            ]
            with open("parsed-md-chunks.json", "w", encoding="utf-8") as f:
                json.dump(saveout, f, indent=2, ensure_ascii=False)

        if len(errored_docs) > 0:
            self.logger.warning(f"Chunking ran into an error in {len(errored_docs)} case(s). List of docs:")
            for ed in errored_docs:
                self.logger.warning(ed)

        return return_docs