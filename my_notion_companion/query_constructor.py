from typing import Dict, Any, List, Sequence, Union, Tuple
from utils import fix_qwen_padding

from langchain_core.runnables import RunnableLambda
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer


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

        with open(self.config["template"]["query_constructor"], "r") as f:
            query_constructor_template = "\n".join(f.readlines())

        self.prompt = PromptTemplate(
            template=query_constructor_template,
            input_variables=["question"],
        )

        self.chain = self.llm | RunnableLambda(fix_qwen_padding)

    def invoke(self, query: str) -> str:
        prompt_msg = self.prompt.invoke({"question": query}).to_string()
        inputs = {
            "role": "user",
            "content": prompt_msg,
        }
        msg = self.tokenizer.apply_chat_template(
            [inputs], tokenize=False, add_generation_prompt=True
        )
        return self.chain.invoke(msg)
