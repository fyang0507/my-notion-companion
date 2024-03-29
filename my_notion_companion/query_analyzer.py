import tomllib
from typing import Any, Dict

from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnableLambda
from loguru import logger
from transformers import AutoTokenizer

from my_notion_companion.few_shot_constructor import FewShotTemplateConstructor


class QueryAnalyzer:
    """QueryAnalyzer helps break a natural language query into structured 'keywords' and 'domain'.

    The keywords will be used to search in documents' content; domain for search in documents' metadata.
    The QueryAnalyzer is used together with lexical document retriever.
    """

    def __init__(
        self,
        llm: LlamaCpp,
        tokenizer: AutoTokenizer,
        config: Dict[str, Any],
        verbose: bool = False,
    ) -> None:

        self.config = config
        self.llm = llm
        self.verbose = verbose

        self.tokenizer = tokenizer

        with open(self.config["template"]["query_analyzer"], "rb") as f:
            self.query_constructor_template = tomllib.load(f)

        self.template = FewShotTemplateConstructor(
            self.tokenizer, self.query_constructor_template
        )

        self.chain = (
            RunnableLambda(self.template.invoke)
            | self.llm
            | RunnableLambda(self.clean_output)
            | RunnableLambda(self.parse_output)
        )

        logger.info("Initialize Query Analyzer.")

    def clean_output(self, s: str) -> str:
        if self.verbose:
            logger.info(f"Query Analyzer output: {s}")

        return s.split("\n\n")[0].split("|")

    def parse_output(self, s: str) -> Dict[str, str]:
        try:
            keywords = (
                s[0]
                .replace(
                    self.query_constructor_template["keyword_prefix"], ""
                )  # remove prefix
                .split(" ")  # keywords are separated with space
            )
            domains = (
                s[1]
                .replace(
                    self.query_constructor_template["domain_prefix"], ""
                )  # remove prefix
                .split(" ")  # domains are separated with space
            )

            if self.verbose:
                logger.info(
                    f"\nQuery Analyzer output\nkeyword: {keywords}\nsearch domains:{domains}"
                )

            return {
                "keywords": keywords,
                "domains": domains,
            }
        except:
            raise QueryConstructorException

    def invoke(self, query: str) -> str:
        return self.chain.invoke(query)


class QueryConstructorException(RuntimeError):
    """Can't construct query error."""
