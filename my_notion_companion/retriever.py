import re
import time
from typing import Any, Dict, List

import jieba
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Redis
from langchain_core.documents.base import Document
from loguru import logger
from thefuzz import fuzz
from transformers import AutoTokenizer

from my_notion_companion.document_metadata_filter import DocumentMetadataFilter
from my_notion_companion.query_analyzer import QueryAnalyzer
from my_notion_companion.utils import load_test_cases


class RedisRetriever:
    def __init__(
        self,
        config: Dict[str, Any],
        token: Dict[str, str],
    ) -> None:

        # init embedding model
        self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key=token["huggingface"], model_name=config["embedding_model"]
        )
        self.vs = Redis.from_existing_index(
            embedding=self.embedding_model,
            index_name=config["index_name"],
            redis_url=config["redis_url"],
            schema=config["redis_schema"],
        )
        self.retriever = self.vs.as_retriever()

    def invoke(self, query):
        return self.retriever.invoke(query)


class BM25SelfQueryRetriever:
    def __init__(
        self,
        llm: LlamaCpp,
        tokenizer: AutoTokenizer,
        docs: List[Document],
        config: Dict[str, Any],
        metadata_match_threshold: float = 0.8,
        content_match_threshold: float = 0.3,
    ) -> None:

        self.llm = llm
        self.tokenizer = tokenizer
        self.config = config
        self.docs = docs

        self.metadata_match_threshold = metadata_match_threshold
        self.content_match_threshold = content_match_threshold

        self._split_documents()

        self.doc_filter = DocumentMetadataFilter(
            self.splits, threshold=self.metadata_match_threshold
        )
        self.query_analyzer = QueryAnalyzer(llm, tokenizer, config, verbose=True)

    def _split_documents(self) -> None:
        rc_splitter = RecursiveCharacterTextSplitter(**self.config["splitter"])
        self.splits = rc_splitter.split_documents(self.docs)

    def _get_relevant_documents(
        self, query: str, splits: List[Document]
    ) -> List[Document]:

        retriever = BM25Retriever.from_documents(
            splits,
            k=self.config["bm25"]["k"],
            preprocess_func=lambda x: jieba.lcut_for_search(x, HMM=False),
        )

        docs_retrieved = retriever.invoke(query)
        docs_filtered = list(
            filter(
                lambda x: fuzz.partial_ratio(x.page_content, query)
                >= self.content_match_threshold * 100,
                docs_retrieved,
            )
        )

        return docs_filtered

    def _filter_documents(self, query_formatted: Dict[str, str]) -> List[Document]:
        keywords: List[str] = query_formatted.get("keywords")
        domains: List[str] = query_formatted.get("domains")

        if len(domains) == 1 and domains[0] == "无":
            logger.info("No filters found by query analyzer.")
            splits = self.splits
        else:
            logger.info(f"filter found by query analyzer: {domains}")
            try:
                splits = self.doc_filter.filter_multiple_criteria(domains, operand="OR")
            except RuntimeError:
                logger.info(
                    "No matched doc found based on query analyzer. Search all docs."
                )
                splits = self.splits

        splits_matched = list()
        for keyword in keywords:
            splits_matched.extend(self._get_relevant_documents(keyword, splits))

        return splits_matched

    def invoke(self, query: str) -> List[Document]:
        try:
            query_formatted = self.query_analyzer.invoke(query)
        except RuntimeError:
            logger.error(
                f"Failed to construct query for the input: {query}, returning raw input."
            )
            return self._get_relevant_documents(query, self.splits)

        return self._filter_documents(query_formatted)


class RetrieverEvaluator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.test_cases = load_test_cases(config["test_path"])

    def evaluate(self, retriever):
        score_list = list()

        for case in self.test_cases:
            time.sleep(0.1)
            docs_retrieved = retriever.invoke(case["q"])
            for ref in case["docs"]:
                score_list.append(self._match_chinese(ref, docs_retrieved))

        logger.info(f"Test cases pass rate: {sum(score_list)/len(score_list):.3f}")
        return score_list

    @staticmethod
    def _match_chinese(string_to_match: str, docs: List[Document]):
        """Check whether Chinese characters in string_to_match appear in at least one doc."""
        # find all chinese characters
        # ref: https://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex
        for tokens in re.findall(r"[\u4e00-\u9fff]+", string_to_match):
            return any([tokens in x.page_content for x in docs])
