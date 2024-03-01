from typing import Dict
from transformers import AutoTokenizer
from transformers.pipelines.conversational import Conversation


class FewShotTemplateConstructor:

    def __init__(self, tokenizer: AutoTokenizer, template: Dict[str, str]):
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
                    "content": self._add_prefix(
                        self.template["user_prefix"], example["user"]
                    ),
                }
            )
            self.history.add_message(
                {
                    "role": "assistant",
                    "content": self._add_prefix(
                        self.template["assistant_prefix"], example["assistant"]
                    ),
                }
            )

    def convert_message_to_llm_format(self, conversation: Conversation):
        # https://huggingface.co/docs/transformers/chat_templating
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _add_prefix(prefix: str, msg: str) -> str:
        return "<< " + prefix + " >>\n" + msg

    def invoke(self, query: str) -> str:
        self.history.add_message(
            {
                "role": "user",
                "content": self._add_prefix(self.template["user_prefix"], query),
            }
        )
        prompt = self.convert_message_to_llm_format(self.history)

        return prompt + self._add_prefix(self.template["assistant_prefix"], "")

    def _check_template(self, template: Dict[str, str]) -> None:
        for key in ["system", "user_prefix", "assistant_prefix", "example"]:
            assert key in template, f"missing key: {key}"

        for example in template["example"]:
            for key in ["user", "assistant"]:
                assert key in example, f"missing key: {key} in {example}"

    def to_string(self) -> str:
        return self.convert_message_to_llm_format(self.history)
