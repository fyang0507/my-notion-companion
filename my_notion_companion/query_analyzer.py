from typing import Dict, Any
import tomllib
from langchain_core.runnables import RunnableLambda
from langchain_community.llms import LlamaCpp
from transformers import AutoTokenizer
from few_shot_constructor import FewShotTemplateConstructor
from loguru import logger


class QueryAnalyzer:
    def __init__(
        self,
        llm: LlamaCpp,
        config: Dict[str, Any],
        verbose: bool = False,
    ) -> None:

        self.config = config
        self.llm = llm
        self.verbose = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"], trust_remote_code=True
        )

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

    @staticmethod
    def clean_output(s: str) -> str:
        return s.split("\n\n")[0].split("|")

    def parse_output(self, s: str, verbose: bool = False) -> Dict[str, str]:
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
    """Can't construct query."""

    pass
