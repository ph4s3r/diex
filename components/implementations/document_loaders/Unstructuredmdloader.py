from components.interfaces.all_interfaces import DocumentLoader

import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_from_base64_gzipped_json
from termcolor import cprint
from langchain_core.documents import Document
import re
import pprint


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
        md_meta["source"] = self.project + self.version + str(md_file.relative_to(self.file_path.parent.parent)).replace("\\", "/")

        elements = partition_md(
            filename=md_file
            )
        
        md_h_list = [""] * 6 # markdown headers have a max depth of 6

        # all microsoft learn docs start with a special header, we process these 3 first elements automatically
        if any(substring in self.project for substring in ["azure", "microsoft"]):           
            try:
                md_meta.update(self._text_to_kv(elements.pop(0).text))
                try:
                    md_meta["intent"] = elements.pop(0).text.split(":")[1:][0]
                except (IndexError, AttributeError):
                    try:
                        md_meta["intent"] = elements.pop(0).text
                    except:
                        raise Exception("Could not process intent, we skip then the header processing")
                md_meta["main_header"] = elements.pop(0).text
            except Exception as e:
                self.logger.warning(f"Skipping processing Microsoft markdown header of {self.file_path}: {e}")
        
        chunks = chunk_by_title(
            elements, 
            combine_text_under_n_chars=0,
            include_orig_elements=True, # stores the elements under ["orig_elements"]
            max_characters=5000
            )
        
        cprint(f"created {len(chunks)} chunks from {md_meta["source"]}", "cyan")

        md_h_list[0] = md_meta.get("main_header", "")
        for chunk in chunks:
            chunk_inmeta = chunk.metadata.to_dict()
            orig_elements = elements_from_base64_gzipped_json(chunk_inmeta["orig_elements"])
            # standalone header 
            # len(orig_elements[0].text) > 5 BUG!!!!!!!!!!!!!!!!!!!!!!!!!!
            if len(orig_elements) == 1 and orig_elements[0].category == 'Title' and len(orig_elements[0].text) > 5:
                # will jnot be saved as a separate chunk / vector - we just save it as a header (metadata)
                depth = orig_elements[0].metadata.category_depth
                # resetting the md headers until the current title depth (the depth 2 header resets an earlier depth 3 and below)
                md_h_list[depth:] = [""] * (len(md_h_list) - depth)
                md_h_list[orig_elements[0].metadata.category_depth] = orig_elements[0].text
                self.logger.debug(f"Standalone chunk: {orig_elements[0].text}")
            else:
                chunk_meta = md_meta
                # all the markdown headers have a category depth = no. of hashtags - 1 - load them into the chunk meta
                for elem in orig_elements:
                    if elem.category == 'Title':
                        depth = elem.metadata.category_depth
                        md_h_list[depth:] = [""] * (len(md_h_list) - depth)
                        md_h_list[elem.metadata.category_depth] = elem.text
                chunk_meta.pop('author', None)
                chunk_meta.pop('ms.author', None)
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
