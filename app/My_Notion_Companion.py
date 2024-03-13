"""
Ref: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app
"""

import random
import time
import tomllib
from pathlib import Path

import streamlit as st
from langchain_community.llms import LlamaCpp
from loguru import logger
from transformers import AutoTokenizer

from my_notion_companion.notion_chatbot import NotionChatBot


def _init_llm(config_path: str) -> NotionChatBot:
    with open(config_path, "rb") as f:
        _CONFIGS = tomllib.load(f)

    llm = LlamaCpp(
        model_path=_CONFIGS["model_path"]
        + "/"
        + _CONFIGS["model_mapping"][_CONFIGS["model_name"]],
        name=_CONFIGS["model_name"],
        **_CONFIGS["llm"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        _CONFIGS["model_name"], trust_remote_code=True
    )

    return NotionChatBot(llm, tokenizer, config_path, verbose=True)


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


class TestInvoke:
    def __init__(self):
        logger.info("initialized")
        self.counter = 0

    def invoke(self):
        self.counter += 1
        return self.counter


def main():

    # add chatbot as a stateful object to session_state
    if "chatbot" not in st.session_state:
        # st.session_state.t = TestInvoke()
        st.session_state.chatbot = _init_llm(Path(__file__).parents[1] / ".config.toml")

    st.set_page_config(
        page_title="My Notion Companion",
        page_icon="🤖",
    )

    st.title("My Notion Companion 🤖")
    st.caption(
        "A conversational RAG that helps to chat with my (mostly Chinese-based) Notion Databases."
    )
    st.caption(
        "Powered by: [🦜🔗](https://www.langchain.com/), [🤗](https://huggingface.co/), [LlamaCpp](https://github.com/ggerganov/llama.cpp), [Streamlit](https://streamlit.io/)."
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Welcome to My Notion Companion. 😉"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # response = st.session_state.t.invoke()
            response = st.session_state.chatbot.invoke(prompt)
            # response = st.write_stream(response_generator())
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
