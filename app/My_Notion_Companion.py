"""
Ref: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app
"""

import tomllib
from pathlib import Path
from typing import Dict

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


def welcome_message() -> Dict[str, str]:
    return {
        "role": "assistant",
        "content": f"Welcome to My Notion Companion. There are in total **{len(st.session_state.chatbot.docs)}** documents available for QA. What's on your mind?",
    }


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
        st.session_state.chatbot = _init_llm(Path(__file__).parents[1] / ".config.toml")

    if "t" not in st.session_state:
        st.session_state.t = TestInvoke()

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
        st.session_state.messages = [welcome_message()]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Two buttons to control history/memory
    def start_over():
        st.session_state.messages = [
            {"role": "assistant", "content": "Okay, let's start over."}
        ]
        st.session_state.chatbot.clear()

    st.sidebar.button(
        "Start All Over Again", on_click=start_over, use_container_width=True
    )

    def clear_chat_history():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Retrieved documents are still in my memory. What else you want to know?",
            }
        ]
        try:
            st.session_state.chatbot.conversational_rag.clear()
        except AttributeError:
            # in case user clicked the button before entering any questions
            # when conversational_rag hasn't been initialized
            pass

    st.sidebar.button(
        "Keep Retrieved Docs but Clear Chat History",
        on_click=clear_chat_history,
        use_container_width=True,
    )

    # Accept user input
    if prompt := st.chat_input("Any questiones?"):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.chatbot.n_query == 0:
            with st.chat_message("assistant"):
                st.write("Retrieving relevant documents. This could take up to 1 min.")

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # response = st.session_state.t.invoke()
            response = st.session_state.chatbot.invoke(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
