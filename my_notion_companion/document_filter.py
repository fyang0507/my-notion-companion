from multiprocessing.pool import ThreadPool
from typing import List

from langchain_core.documents.base import Document
from loguru import logger
from thefuzz import fuzz


class DocumentFilter:
    def __init__(
        self,
        docs: List[Document],
        threshold: float,
    ):
        assert threshold >= 0.0 and threshold <= 1.0, "threshold must be between [0,1]."
        logger.info(f"Setting metadata fuzzy match threshold to: {threshold}.")
        self.threshold = threshold
        self.docs = docs

    def _find_match(self, val: str, doc_list: List[Document]) -> List[int]:
        def _filter_func(doc: Document) -> bool:
            has_match = False
            for attr in doc.metadata:
                # the search value must be both qualified for partial and full match
                if (
                    fuzz.partial_ratio(str(doc.metadata[attr]), val)
                    >= self.threshold * 100
                ) and (
                    fuzz.ratio(str(doc.metadata[attr]), val) >= self.threshold * 100
                ):
                    has_match = True
                    break
            return has_match

        t = ThreadPool()
        is_matched = t.map(_filter_func, doc_list)
        t.close()
        return is_matched

    def filter(self, val: str) -> List[Document]:
        is_matched = self._find_match(val, self.docs)
        docs = [x for x, b in zip(self.docs, is_matched) if b]

        if docs:
            logger.info(f"Remaining doc: {len(docs)/len(self.docs): .3f}")
            return docs
        else:
            raise NoMatchedDocException(
                "No matched docs based on input filter criteria."
            )

    def filter_multiple_criteria(
        self, criteria: List[str], operand: str = "OR"
    ) -> List[Document]:
        assert operand in ["OR", "AND"], "operand has to be either OR or AND."

        if operand == "AND":
            docs = self.docs
            for c in criteria:
                result = self._find_match(c, docs)
                docs = [x for x, i in zip(docs, result) if i]

        elif operand == "OR":
            is_matched = [False] * len(self.docs)
            for c in criteria:
                result = self._find_match(c, self.docs)
                is_matched = [x or y for x, y in zip(is_matched, result)]
            docs = [x for x, i in zip(self.docs, is_matched) if i]

        if not docs:
            raise NoMatchedDocException(
                "No matched docs based on input filter criteria."
            )

        logger.info(f"Remaining doc: {len(docs)/len(self.docs): .3f}")
        return docs


class NoMatchedDocException(RuntimeError):
    """No matched docs based on input filter criteria."""

    pass
