import tomllib

from langchain_community.llms import LlamaCpp
from loguru import logger
from transformers import AutoTokenizer

from my_notion_companion.conversational_rag import ConversationalRAG
from my_notion_companion.document_match_checker import DocumentMatchChecker
from my_notion_companion.document_metadata_filter import NoMatchedDocException
from my_notion_companion.retriever import BM25SelfQueryRetriever, RedisRetriever
from my_notion_companion.utils import load_notion_documents, peek_docs


class NotionChatBot:
    """NotionChatBot Chains document retriever and conversational RAG."""

    def __init__(
        self,
        llm: LlamaCpp,
        tokneizer: AutoTokenizer,
        config_path: str,
        verbose: bool = False,
    ) -> None:

        # load from configuration files
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)
        with open(self.config["path"]["tokens"], "rb") as f:
            self.tokens = tomllib.load(f)
        with open(self.config["template"]["conversatoinal_rag"], "rb") as f:
            self.system_message = tomllib.load(f)["system"]

        self.llm = llm
        self.tokenizer = tokneizer
        self.verbose = verbose

        self.docs = load_notion_documents(self.config, self.tokens)
        self._initialize_retriever()

        self.n_query = 0

    def clear(self) -> None:
        """Clear history and retrieved documents."""
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

                self.conversational_rag = ConversationalRAG(
                    self.llm,
                    self.tokenizer,
                    self.config,
                    self.system_message,
                    docs_retrieved,
                )
                logger.info("Initialize Conversational RAG.")
            else:
                raise NoMatchedDocException()

        return self.conversational_rag.invoke(query)

    def _initialize_retriever(self) -> None:
        # initialize lexical BM25 retriever
        self.retriever_lexical = BM25SelfQueryRetriever(
            self.llm, self.tokenizer, self.docs, self.config
        )
        # Initialize semantic Redis retriever
        self.retriever_semantic = RedisRetriever(self.config, self.tokens)
        # Initialize post-retrieval document relevance checker
        self.match_checker = DocumentMatchChecker(
            self.llm, self.tokenizer, self.config, self.verbose
        )
