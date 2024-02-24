
from typing import List
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language

from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

class JavaScriptFileLoader(BaseLoader):
    """
    A document loader for JavaScript files that uses the LanguageParser to parse the file.
    The main advantage of using a language parser in addition to a blob loader are:
        - Keep top-level functions and classes together (into a single document)
        - Put remaining code into a separate document
        - Retains metadata about where each split comes from
    """
    def __init__(self, path: str, parser_threshold: int = 500) -> None:
        """_summary_
        Initialize the JavaScript File Loader
        Args:
            path (str): _description_  The absolute path to the JavaScript file as string.
            parser_threshold (int, optional): _description_. Defaults to 500.
        """
        self.path = path
        self.parser_threshold = parser_threshold
        self.parser = LanguageParser(language=Language.JS, parser_threshold=self.parser_threshold)

    def load(self) -> List[Document]:
        blob: Blob = Blob.from_path(self.path)
        return self.parser.parse(blob)