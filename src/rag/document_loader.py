"""
src/document_loader.py

ECBSpeechLoader:
    load_reports
    create_documents

"""

import json
from typing import Dict, List
from langchain_core.documents import Document

class ECBSpeechLoader:

    """
    A class to handle ECB speeches downloaded from the ECB API.
    Speeches are already in json format.

    Handles loading from json, create list, and stores metadata
    
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.reports = []

    def load_reports(self) -> List[Dict]:
        """Load ECB speeches from json file.
        """
        # we first load the reports and raise errors if not found
        try:
            with open(self.data_path, "r") as f:
                self.reports = json.load(f)

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {self.data_path}")
        return self.reports
    


    def create_documents(self) -> List[Document]:
        """Convert ECB  reports to LangChain documents.
        Document content and metadata  contain:
        1. speech content
        2. metadata : date, speakers, subtitle, source
        """

        #check if report have been loaded
        if not self.reports:

            self.load_reports()

        print(f"DEBUG create_documents — {len(self.reports)} reports")

        documents = [] # initialise empty list of docuemnts

        for report in self.reports:

            # Build readable content for embeddings
            # the structures are simillar so we overspecificy the type of report
            content = f"""

Title: {report["metadata"]["title"]} 
Speaker: {report["metadata"]['speaker']}
Publication Date: {report["metadata"]['date']}
Source: {report["metadata"]['source']}

Content:
{report['page_content']}
"""

            # we keep metadata separate for filtering and citation
            metadata = report["metadata"] 

            
            # we need to append the documents
            documents.append(Document(page_content=content, metadata=metadata))

            print(f"DEBUG create_documents — {len(documents)} documents created")

        return documents




