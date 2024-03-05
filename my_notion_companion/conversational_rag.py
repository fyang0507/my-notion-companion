from typing import Dict, Any, List

from langchain_community.llms import LlamaCpp

from transformers import AutoTokenizer
from transformers.pipelines.conversational import Conversation
from langchain_core.prompt_values import StringPromptValue
from utils import fix_qwen_padding

from langchain_core.documents.base import Document


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
        return "\n\n".join(doc.page_content for doc in docs)

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

        inputs = self.convert_message_to_llm_format(self.conversation)

        response = self.llm.invoke(inputs)

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

        # prevent memory overflow
        self._keep_k_rounds_most_recent_conversation()

        return response

    def clear_conversation(self):
        self.conversation = Conversation()

    def _keep_k_rounds_most_recent_conversation(self):
        k = self.model_params["conversation"]["k_rounds"]
        if len(self.conversation) > 2 * k:
            # keep if system input exists
            if self.conversation[0]["role"] == "system":
                self.conversation = Conversation(
                    [self.conversation[0]] + self.conversation[-2 * k :]
                )
            else:
                self.conversation = Conversation(self.conversation[-2 * k :])

    def extract_ai_responses(self):
        return self.full_history.generated_responses
