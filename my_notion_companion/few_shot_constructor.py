from typing import Dict

from transformers import AutoTokenizer
from transformers.pipelines.conversational import Conversation

from my_notion_companion.utils import convert_message_to_llm_format


class FewShotTemplateConstructor:
    """Few shot template constructor.

    Uses Hugging Face's generalizable chat template API.
    """

    def __init__(self, tokenizer: AutoTokenizer, template: Dict[str, str]) -> None:
        self._check_template(template)
        self.template = template

        self.tokenizer = tokenizer
        self.construct_history()

    def construct_history(self):
        self.history = Conversation()

        # add system message
        self.history.add_message({"role": "system", "content": self.template["system"]})

        for example in self.template["example"]:
            self.history.add_message(
                {
                    "role": "user",
                    "content": example["user"],
                }
            )
            self.history.add_message(
                {
                    "role": "assistant",
                    "content": example["assistant"],
                }
            )

    def invoke(self, query: str) -> str:
        self.history.add_message(
            {
                "role": "user",
                "content": query,
            }
        )
        prompt = convert_message_to_llm_format(self.tokenizer, self.history)
        self.history = Conversation(
            self.history[:-1]
        )  # remove the input query so it doesn't pile up after the few-shot examples
        return prompt

    def _check_template(self, template: Dict[str, str]) -> None:
        for key in ["system", "example"]:
            assert key in template, f"missing key: {key}"

        for example in template["example"]:
            for key in ["user", "assistant"]:
                assert key in example, f"missing key: {key} in {example}"

    def to_string(self) -> str:
        return convert_message_to_llm_format(self.tokenizer, self.history)
