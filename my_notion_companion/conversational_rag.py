from typing import Any, Dict, List

from langchain_community.llms import LlamaCpp
from langchain_core.documents.base import Document
from langchain_core.prompt_values import StringPromptValue
from loguru import logger
from transformers import AutoTokenizer
from transformers.pipelines.conversational import Conversation

from my_notion_companion.utils import convert_message_to_llm_format, format_docs


class ConversationalRAG:
    """ConversationalRAG class is the entry point for 'chat with documents'.

    It expect a list of documents as initialize param, and will construct system message with the documents.
    It will have a memory of the chat history and user can ask follow up questions.
    """

    def __init__(
        self,
        llm: LlamaCpp,
        tokenizer: AutoTokenizer,
        config: Dict[str, Any],
        system_message: str,
        contexts: List[Document],
    ):
        self.llm = llm
        self.tokenizer = tokenizer

        self.model_params = config["llm"]
        self.k_rounds_memory = self.model_params["conversation"]["k_rounds"]

        # initialize memory
        self.conversation = Conversation()
        self.full_history = Conversation()

        # initialize system message
        self.system_message = {
            "role": "system",
            "content": system_message + "\n\n" + format_docs(contexts),
        }

        # add system message to the conversation history
        self.conversation.add_message(self.system_message)
        self.full_history.add_message(self.system_message)

    def invoke(self, text: str) -> str:
        # convert StringPromptValue (str like object but with format check) to string
        # it is incomptible with tokenizer.apply_chat_template()
        # ref: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompt_values.py
        if isinstance(text, StringPromptValue):
            text = text.to_string()

        # format input text
        inputs = {
            "role": "user",
            "content": text,
        }

        # add the message to the memeory
        self.conversation.add_message(inputs)
        self.full_history.add_message(inputs)

        # in case of context window overflow, reduce the memory size by keeping
        # only the last k rounds of conversations
        try:
            logger.info(f"Sending query to conversatonal RAG: {text}")
            response = self.llm.invoke(
                convert_message_to_llm_format(self.tokenizer, self.conversation)
            )
        except ValueError:
            logger.info(
                f"Prompt token exceed context window of {self.llm.n_ctx}.\n"
                f"Clear memory and keep only the last {self.k_rounds_memory} rounds of conversation."
            )
            # remove old conversations form memory
            self._keep_k_rounds_most_recent_conversation()
            logger.info(f"Retry: sending query to conversatonal RAG: {text}")
            response = self.llm.invoke(
                convert_message_to_llm_format(self.tokenizer, self.conversation)
            )

        logger.info(f"Received response: {response}")
        # invoke chain and format to Conversation-style response
        response_dict = {
            "role": "assistant",
            "content": response,
        }

        # add response to memory
        self.conversation.add_message(response_dict)
        self.full_history.add_message(response_dict)

        return response

    def clear(self) -> None:
        """Clear conversation memory."""
        logger.info(
            "Clear conversation history. Please re-enter the follow up questions."
        )
        self.conversation = Conversation()
        self.full_history = Conversation()
        # add system message to the conversation history
        self.conversation.add_message(self.system_message)
        self.full_history.add_message(self.system_message)

    def _keep_k_rounds_most_recent_conversation(self) -> None:
        """Keep only the last K rounds of conversation in memory.

        Avoid memory overflow because of context size limit.
        """
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
        """Retrieve only assistant responses."""
        return self.full_history.generated_responses
