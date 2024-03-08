from typing import Any, Dict, List

from langchain_community.llms import LlamaCpp
from langchain_core.documents.base import Document
from langchain_core.prompt_values import StringPromptValue
from loguru import logger
from transformers import AutoTokenizer
from transformers.pipelines.conversational import Conversation
from utils import fix_qwen_padding


class ConversationalRAG:
    def __init__(
        self,
        llm: LlamaCpp,
        config: Dict[str, Any],
        system_message: str,
        contexts: List[Document],
    ):
        self.llm = llm

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"], trust_remote_code=True
        )

        self.model_params = config["llm"]
        self.k_rounds_memory = self.model_params["conversation"]["k_rounds"]

        self.conversation = Conversation()
        self.full_history = Conversation()

        self.sys_msg = {
            "role": "system",
            "content": system_message + "\n\n" + self._format_docs(contexts),
        }

        # add system message to the conversation history
        self.conversation.add_message(self.sys_msg)
        self.full_history.add_message(self.sys_msg)

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        def format_doc_with_metadata(doc: Document) -> str:
            return "内容：\n" + doc.page_content + "\n\n元数据：\n" + str(doc.metadata)

        s = ""
        for idx, doc in enumerate(docs):
            s += f"文档{idx+1}\n\n"
            s += format_doc_with_metadata(doc)
            s += "\n\n\n"

        return s[:-3]  # skip the last \n\n\n

    def convert_message_to_llm_format(self, msg: Conversation):
        # https://huggingface.co/docs/transformers/chat_templating
        return self.tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )

    def invoke(self, text: str) -> Dict[str, str]:

        # convert StringPromptValue (str like object but with format check) to string
        # it is incomptible with tokenizer.apply_chat_template()
        # ref: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompt_values.py
        if isinstance(text, StringPromptValue):
            text = text.to_string()

        inputs = {
            "role": "user",
            "content": text,
        }

        # add the message to the memeory
        self.conversation.add_message(inputs)
        self.full_history.add_message(inputs)

        try:
            response = self.llm.invoke(
                self.convert_message_to_llm_format(self.conversation)
            )
        except ValueError:
            logger.info(
                f"Prompt token exceed context window of {self.llm.n_ctx}.\n"
                f"Clear memory and keep only the last {self.k_rounds_memory} rounds of conversation."
            )
            # remove old conversations form memory
            self._keep_k_rounds_most_recent_conversation()
            response = self.llm.invoke(
                self.convert_message_to_llm_format(self.conversation)
            )

        if self.llm.name == "Qwen/Qwen-7B-Chat":
            response = fix_qwen_padding(response)

        # invoke chain and format to Conversation-style response
        response = {
            "role": "assistant",
            "content": response,
        }

        # add response to memory
        self.conversation.add_message(response)
        self.full_history.add_message(response)

        return response

    def clear(self):
        self.conversation = Conversation()
        self.full_history = Conversation()
        # add system message to the conversation history
        self.conversation.add_message(self.sys_msg)
        self.full_history.add_message(self.sys_msg)

    def _keep_k_rounds_most_recent_conversation(self) -> None:
        if self.k_rounds_memory < 0:
            return
        elif len(self.conversation) > 2 * self.k_rounds_memory:
            # keep if system input exists
            if self.conversation[0]["role"] == "system":
                self.conversation = Conversation(
                    [self.conversation[0]]
                    + self.conversation[-2 * self.k_rounds_memory + 1 :]
                )
            else:
                self.conversation = Conversation(
                    self.conversation[-2 * self.k_rounds_memory + 1 :]
                )

    def extract_ai_responses(self):
        return self.full_history.generated_responses
