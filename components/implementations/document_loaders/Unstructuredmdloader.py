from components.interfaces.all_interfaces import DocumentLoader

import re
import logging
import pprint
from pathlib import Path
from termcolor import cprint
from typing import List, Optional
from langchain_core.documents import Document
from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_from_base64_gzipped_json
from langchain_core.documents import Document




class UnstructuredMDLoader(DocumentLoader):
    def __init__(
            self, 
            file_path: str, 
            max_workers: Optional[int] = None, 
            project: str = "",
            version: str = ""
            ) -> None:
        """
        Initializes the MarkdownLoader with a directory path and optional max_workers for parallel loading.
        https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document
        https://python.langchain.com/docs/integrations/document_loaders/unstructured_markdown/
        https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html

        Args:
            file_path (str): The path to the directory containing Markdown files.
            max_workers (Optional[int]): The maximum number of threads to use for parallel loading.
                                        Defaults to the number of processors on the machine, multiplied by 5.
        """
        self.project = project
        self.version = version
        self.file_path: Path = Path(file_path).resolve()
        self.logger: logging.Logger = logging.getLogger('DocumentLoader')
        self.debug = False


    def load(self) -> List[Document]:
        """
        Recursively loads all Markdown files from the directory and returns them as a list of Document objects.

        Returns:
            List[Document]: A list of loaded documents.
        """
        docs = []

        try:
            if not self.file_path.exists():
                self.logger.error(f"Directory does not exist: {self.file_path}")
                return []
            if not self.file_path.is_dir():
                self.logger.error(f"Provided path is not a directory: {self.file_path}")
                return []
        except Exception as e:
            self.logger.error(f"Unhandled exception loading directory {self.file_path}: {e}")

        # Recursively find all .md files
        markdown_files = list(self.file_path.rglob('*.md'))
        num_files = len(markdown_files)
        self.logger.info(f"Found {num_files} Markdown file(s) in {self.file_path}")
        

        for mdfile in markdown_files:
            mdelements = self._unstruct_partition_single_md(mdfile)
            if mdelements is not None:
                docs.extend(mdelements) # lehet extend kell
        return docs
    
    def _text_to_kv(self, input_string):
        """
        Parses the input string into a dictionary of key-value pairs.
        
        Args:
            input_string (str): The string containing key-value pairs separated by colons.
            
        Returns:
            dict: A dictionary containing all extracted key-value pairs.
        """
        # Regular expression pattern to match key-value pairs
        pattern = r'(\b[\w\.]+):\s*([^:]+)(?=\s+\b[\w\.]+:|$)'
        
        # Find all matches in the input string
        matches = re.findall(pattern, input_string)
        
        # Construct the dictionary from the matches
        kv_pairs = {key.strip(): value.strip() for key, value in matches}
        
        return kv_pairs
    

    def _unstruct_partition_single_md(self, md_file: Path) -> Optional[List[Document]]:

        # this is what we return
        result_document_list = []

        md_meta = {} # all the custom metadata we gather manually from the docs
        # warning, this need to be checked every time
        md_meta["source"] = self.project + self.version + str(md_file.relative_to(self.file_path.parent.parent)).replace("\\", "/")
        try:
            elements = partition_md(
                filename=md_file
                )
        except Exception as e:
            self.logger.error(f"Skipping processing markdown file of {md_meta["source"]}: {e}")
            return None
        # all microsoft learn docs start with a special header, we try to process these 3 first elements automatically
        if any(substring in self.project for substring in ["azure", "microsoft"]) and "includes" not in md_meta["source"]:           
            try:
                msheaders = self._text_to_kv(elements[0].text)
                elements.pop(0).text
                md_meta.update(msheaders)
            except Exception as e:
                self.logger.warning(f"Skipping processing Microsoft markdown header of {md_meta["source"]}: {e}")
            if len(elements) > 0:
                if "intent" in elements[0].text and elements[0].category == "Title":
                    try:
                        intent = elements[0].text.split(":")[1:][0]
                        md_meta["intent"] = intent
                        elements.pop(0)
                    # then try another way
                    except:
                        try:
                            md_meta["intent"] = elements[0].text
                            elements.pop(0)
                        except Exception as e:
                            self.logger.warning(f"Could not process intent, we skip then the header processing, {e}")
                        
                # no intent doc, lets try with processing the first header if there is no intent
                if elements[0].category == "Title":
                    md_meta["main_header"] = elements.pop(0).text
            else:
                self.logger.warning(f"Not much stuff in here, skipping {md_meta["source"]}")

        
        chunks = chunk_by_title(
            elements, 
            combine_text_under_n_chars=0,
            include_orig_elements=True, # stores the elements under ["orig_elements"]
            max_characters=5000
            )
        
        self.logger.debug(f"created {len(chunks)} chunks from {md_meta["source"]}", "cyan")

        for chunk in chunks:
            md_h_list = [""] * 6 # markdown headers have a max depth of 6
            md_h_list[0] = md_meta.get("main_header", "")
            chunk_meta = md_meta
            chunk_inmeta = chunk.metadata.to_dict()
            orig_elements = elements_from_base64_gzipped_json(chunk_inmeta["orig_elements"])
            # standalone header 
            # len(orig_elements[0].text) > 5 BUG!!! :
            # unstructured sometimes classifies a few standalone words or just a random line as a title, while our mechanism here relies on the promised title based splitting...
            if len(orig_elements) == 1 and orig_elements[0].category == 'Title' and len(orig_elements[0].text) > 5:
                # will jnot be saved as a separate chunk / vector - we just save it as a header (metadata)
                depth = orig_elements[0].metadata.category_depth
                # resetting the md headers until the current title depth (the depth 2 header resets an earlier depth 3 and below)
                md_h_list[depth:] = [""] * (len(md_h_list) - depth)
                md_h_list[orig_elements[0].metadata.category_depth] = orig_elements[0].text
                self.logger.debug(f"Standalone chunk: {orig_elements[0].text}")
            else:
                # all the markdown headers have a category depth = no. of hashtags - 1 - load them into the chunk meta
                for elem in orig_elements:
                    if elem.category == 'Title':
                        depth = elem.metadata.category_depth
                        md_h_list[depth:] = [""] * (len(md_h_list) - depth)
                        md_h_list[elem.metadata.category_depth] = elem.text
                # remove empty headers
                while len(md_h_list) > 0:
                    if md_h_list[-1] == "":
                        md_h_list.pop()
                    else:
                        break
                chunk_meta.pop('author', None)
                chunk_meta.pop('ms.author', None)
                if len(md_h_list) > 0:
                    markdown_header_struct_dict = {f"header_{i}": value for i, value in enumerate(md_h_list)}
                    chunk_meta.update(markdown_header_struct_dict)
                result_document_list.append(Document(
                    page_content=str(chunk),
                    metadata=chunk_meta
                ))
                if self.debug:
                    cprint(f"  CHUNK META:", "red")
                    pprint.pprint(chunk_meta)
                    cprint(f"  DOC page_content:", "red")
                    cprint(str(chunk), "green")
                    cprint("\n\n" + "-"*80, "red")

        return result_document_list
