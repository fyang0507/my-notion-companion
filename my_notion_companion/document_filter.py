from langchain_core.documents.base import Document
from typing import List
from thefuzz import fuzz
from loguru import logger
from multiprocessing.pool import ThreadPool


class DocumentFilter:
    def __init__(self, threshold):
        self.threshold = threshold

    def filter_by_metadata(self, val: str, doc_list: List[Document]) -> List[Document]:
        def _filter_func(doc: Document) -> bool:
            has_match = False
            for attr in doc.metadata:
                if (
                    fuzz.partial_ratio(str(doc.metadata[attr]), val)
                    >= self.threshold * 100
                ):
                    has_match = True
                    break
            return has_match

        t = ThreadPool()
        booleans = t.map(_filter_func, doc_list)
        t.close()

        matched_list = [x for x, b in zip(doc_list, booleans) if b]

        logger.info(f"Remaining doc: {len(matched_list)/len(doc_list): .3f}")
        return matched_list

    # TODO: if no matched_list, raise exception
