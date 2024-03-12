import os
import pickle
import tomllib

from conversational_rag import ConversationalRAG
from document_filter import NoMatchedDocException
from document_match_checker import DocumentMatchChecker
from langchain_community.llms import LlamaCpp
from loguru import logger
from notion_loader import NotionLoader
from retriever import BM25SelfQueryRetriever, RedisRetriever
from transformers import AutoTokenizer
from utils import peek_docs


class NotionChatBot:
    def __init__(
        self,
        llm: LlamaCpp,
        tokneizer: AutoTokenizer,
        config_path: str,
        verbose: bool = False,
    ) -> None:

        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        with open(self.config["path"]["tokens"], "rb") as f:
            self.tokens = tomllib.load(f)

        with open(self.config["template"]["conversatoinal_rag"], "rb") as f:
            self.system_message = tomllib.load(f)["system"]

        self.llm = llm
        self.tokenizer = tokneizer
        self.verbose = verbose

        self._load_documents()
        self._initialize_retriever()

        self.n_query = 0

    def clear(self) -> None:
        logger.info("Clear retrieved documents. Please re-enter the prompt.")
        self.n_query = 0

    def invoke(self, query: str) -> str:
        self.n_query += 1

        if self.n_query == 1:
            logger.info("Try lexical search.")
            docs_retrieved = self.retriever_lexical.invoke(query)

            doc_ids = set([x.metadata["id"] for x in docs_retrieved])

            if len(docs_retrieved) <= 2:
                logger.info(
                    f"{len(docs_retrieved)} docs found via lexical search. Try semantic search."
                )
                docs_semantic = self.retriever_semantic.invoke(query)
                logger.info(
                    f"{len(docs_semantic)} docs found via semantic search. Use LLM to check relevance."
                )
                docs_filtered = self.match_checker.invoke(docs_semantic, query)
                docs_retrieved = docs_retrieved + list(
                    filter(lambda x: x.metadata["id"] not in doc_ids, docs_filtered)
                )

            if self.verbose:
                logger.info(f"Retrieved relevant docs:\n\n{peek_docs(docs_retrieved)}")

            if len(docs_retrieved) > 0:

                self.conversatoinal_rag = ConversationalRAG(
                    self.llm,
                    self.tokenizer,
                    self.config,
                    self.system_message,
                    docs_retrieved,
                )
                logger.info("Initialize Conversational RAG.")
            else:
                raise NoMatchedDocException()

        return self.conversatoinal_rag.invoke(query)

    def _load_documents(self) -> None:
        if not self.config["force_repull"] and os.path.exists(
            self.config["path"]["docs"]
        ):
            logger.info("Load data from existing offline copy.")
            with open(self.config["path"]["docs"], "rb") as f:
                docs = pickle.load(f)
        else:
            logger.info("Load data from notion API.")
            with open(self.config["path"]["notion_dbs"], "rb") as f:
                _DATABASES_NOTION = tomllib.load(f)

            loader = NotionLoader(self.tokens, _DATABASES_NOTION)
            loader.export_to_pickle(self.config["path"]["docs"])
            docs = loader.load()

        self.docs = docs

    def _initialize_retriever(self) -> None:

        self.retriever_lexical = BM25SelfQueryRetriever(
            self.llm, self.tokenizer, self.docs, self.config
        )
        self.retriever_semantic = RedisRetriever(self.config, self.tokens)
        self.match_checker = DocumentMatchChecker(
            self.llm, self.tokenizer, self.config, self.verbose
        )
