import pickle
import random
import time
from typing import Any, Dict, List

from langchain_community.document_loaders import NotionDBLoader
from langchain_core.documents.base import Document
from loguru import logger


class NotionLoader:
    """Notion loader.

    Uses Langchain's NotionDBLoader as the backend with additional metadata preprocessing.
    Note: Notion API seems to have a rate limit and parallel call will be refused.
    """

    def __init__(
        self,
        tokens: Dict[str, str],
        notion_dbs: Dict[str, Any],
    ) -> None:

        self._tokens = tokens
        self._notion_dbs = notion_dbs
        self.data_raw = None
        self.data_formatted = None

    def _load_databases(self) -> None:
        logger.info("Read from notion databases.")
        self.data_raw = {}

        # this step could be parallelized
        # however, it seems Notion API has imposed some rate limit, and when parallelized
        # it is easy to get connection errors
        for db_name, db_id in self._notion_dbs.items():
            logger.info(f"retrieving from database: {db_name}")
            loader = NotionDBLoader(
                integration_token=self._tokens["notion"],
                database_id=db_id,
                request_timeout_sec=30,  # optional, defaults to 10
            )

            data = None
            for i in range(3):
                try:
                    logger.info(
                        f"trying {i+1} time, retrieving from database: {db_name}"
                    )
                    data = loader.load()
                except:
                    time.sleep(random.randrange(30) + 5)

                if data:
                    logger.info(f"completed retrieval from database: {db_name}")
                    break

            if data:
                logger.info(f"completed retrieval all databases from Notion")
                self.data_raw[db_name] = data
            else:
                raise RuntimeError(f"Failed to retrieve from database: {db_name}")

    def _format_data(self) -> None:
        """
        1. Transform raw data into a flattened list, adding database source as a metadata field "source"
        2. Process metadata fields
        """
        self.data_formatted = list()

        for db_name, docs in self.data_raw.items():
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

            self.data_formatted.extend(docs)

    def load(self) -> List[Document]:
        self._load_databases()
        self._format_data()
        return self.data_formatted

    def export_to_pickle(self, path_export: str) -> None:
        with open(path_export, "wb") as f:
            pickle.dump(self.data_formatted, f)
