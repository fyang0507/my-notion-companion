"""
Ref: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app
"""

import streamlit as st

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

# st.sidebar.success("Select a demo above.")


import random
import time


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


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
