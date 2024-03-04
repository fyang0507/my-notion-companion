from langchain_community.document_loaders import NotionDBLoader
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from langchain_core.documents.base import Document
from loguru import logger
import pickle
import time
import random


class NotionLoader:

    def __init__(
        self,
        tokens: Dict[str, str],
        notion_dbs: Dict[str, Any],
    ) -> None:

        self._tokens = tokens
        self._notion_dbs = notion_dbs
        self.data_raw = None
        self.data_formatted = None

    def _load_db(self, db_name) -> List[Document]:
        loader = NotionDBLoader(
            integration_token=self._tokens["notion"],
            database_id=self._notion_dbs[db_name],
            request_timeout_sec=30,  # optional, defaults to 10
        )
        # to avoid concurrent request being sent
        time.sleep(random.randrange(10) + 1)
        data = loader.load()
        return data

    def _load_databases(self) -> None:
        logger.info("Read from notion databases.")
        e = ThreadPoolExecutor()
        results = list(e.map(self._load_db, self._notion_dbs.keys()))
        logger.info("Completed reading from notion")
        self.data_raw = dict(zip(self._notion_dbs.keys(), results))

    def _format_data(self) -> None:
        """
        1. Transform raw data into a flattened list, adding database source as a metadata field "source"
        2. Process metadata fields
        """
        docs_list = list()

        for db_name, docs in self.raw_data.items():
            for doc in docs:
                # because our data are gathered from multiple databases
                # we are going to throw the database names as one property
                # into the docs' metadata field
                # and return as a list
                doc.metadata["source"] = db_name

                # change dates into YYYYMMDD int format to allow GT/LT/EQ comparison
                if "date" in doc.metadata:
                    if "start" in doc.metadata["date"]:
                        doc.metadata["date_start"] = int(
                            doc.metadata["date"]["start"].replace("-", "")
                        )
                    if "end" in doc.metadata["date"] and doc.metadata["date"]["end"]:
                        doc.metadata["date_end"] = int(
                            doc.metadata["date"]["end"].replace("-", "")
                        )

                    del doc.metadata["date"]

                if "tags" in doc.metadata:
                    doc.metadata["tags"] = ", ".join(doc.metadata["tags"])

            docs_list.extend(docs)

        self.data_formatted = docs_list

    def load(self) -> List[Document]:
        self._load_databases()
        self._format_data()
        return self.data_formatted

    def export_to_pickle(self, path_export: str) -> None:
        with open(path_export, "wb") as f:
            pickle.dump(self.data_formatted, f)
