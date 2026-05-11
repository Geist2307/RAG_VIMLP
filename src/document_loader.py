import json
from typing import Dict, List
from langchain_core.documents import Document

class FinancialDocumentLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.reports = []

    def load_reports(self) -> List[Dict]:
        """Load financial reports from json file.
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
        """Convert financial reports to LangChain documents.
        Document content and metadata  contain:
        1. title
        2. publication_date 
        3. author
        4. category
        5. summary
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

Title: {report["title"]} 
Category: {report['category']}
Publication Date: {report['publication_date']}
Author: {report['author']}

Summary:
{report['summary']}

Last 5 days for report  for {report['title']}:
{chr(10).join(f"  - {stat}" for stat in report['key_statistics'])}
"""



            # we keep metadata separate for filtering and citation
            metadata = {

                "report_id": report["reportId"],
                "title": report["title"],
                "publication_date" : report["publication_date"],
                "author" : report["author"]


            }

            # we need to append the documents
            documents.append(Document(page_content=content, metadata=metadata))

            print(f"DEBUG create_documents — {len(documents)} documents created")

        return documents




