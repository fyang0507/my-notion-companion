from typing import Dict, Any, List, Sequence, Union, Tuple
import tomllib

from langchain_core.runnables import RunnableLambda
from langchain_community.llms import LlamaCpp
from transformers import AutoTokenizer
from few_shot_constructor import FewShotTemplateConstructor


class QueryConstructor:
    def __init__(
        self,
        llm: LlamaCpp,
        config: Dict[str, Any],
    ) -> None:

        self.config = config
        self.llm = llm

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"], trust_remote_code=True
        )

        with open(self.config["template"]["query_constructor"], "rb") as f:
            query_constructor_template = tomllib.load(f)

        self.template = FewShotTemplateConstructor(
            self.tokenizer, query_constructor_template
        )

        self.chain = (
            RunnableLambda(self.template.invoke)
            | self.llm
            | RunnableLambda(self.clean_output)
        )

    @staticmethod
    def clean_output(s: str) -> str:
        return s.split("\n\n")[0]

    def invoke(self, query: str) -> str:
        return self.chain.invoke(query)
