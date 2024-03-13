import tomllib
from typing import Any, Dict, List

from langchain_community.llms import LlamaCpp
from langchain_core.documents.base import Document
from langchain_core.runnables import RunnableLambda
from loguru import logger
from transformers import AutoTokenizer

from my_notion_companion.few_shot_constructor import FewShotTemplateConstructor
from my_notion_companion.utils import peek_docs


class DocumentMatchChecker:
    """DocumentMatchChecker supports checking whether a doc is relevant to a query.

    It requires an input list of documents and outputs a filtered list of documents.
    """

    def __init__(
        self,
        llm: LlamaCpp,
        tokenizer: AutoTokenizer,
        config: Dict[str, Any],
        verbose: bool = False,
    ) -> None:

        self.llm = llm
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.config = config

        # load few shot template
        with open(self.config["template"]["document_match_checker"], "rb") as f:
            self.query_constructor_template = tomllib.load(f)
        self.template = FewShotTemplateConstructor(
            self.tokenizer, self.query_constructor_template
        )

        # construct the chain
        self.chain = RunnableLambda(self.template.invoke) | self.llm

    def invoke(self, docs: List[Document], query: str) -> List[Document]:
        docs_filtered = list()

        for doc in docs:
            user_content = self._format_user_content(doc, query)
            response = self.chain.invoke(user_content)

            if self.verbose:
                logger.info(f"compare relevance with doc:\n\n{peek_docs([doc])}")
                logger.info(f"conclusion: {response}")

            if self.is_relevant(response):
                docs_filtered.append(doc)

        return docs_filtered

    @staticmethod
    def _format_user_content(doc: Document, query: str) -> str:
        return f"""<< 资料1 >>\n{query}\n\n<< 资料2 >>\n{doc.page_content}"""

    def is_relevant(self, response: str) -> bool:
        # use a loose definition
        # can't prevent the llm from ouputting reasoning even with few-shot examples
        if "不相关" in response:
            return False
        return True
