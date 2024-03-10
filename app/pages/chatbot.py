"""
Ref: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
"""

import streamlit as st

st.title("🤖 My Notion Campnion")
st.caption(
    "A conversational RAG that helps answer questions of my (mostly Chinese-based) Notion Databases."
)
st.caption(
    "Powered by: [🦜🔗](https://www.langchain.com/), [🤗](https://huggingface.co/), [LlamaCpp](https://github.com/ggerganov/llama.cpp), [Streamlit](https://streamlit.io/)."
)


st.markdown(
    """
Search func in Notion sucks. It supports only discrete keyword search with exact match (e.g. it treats Taylor Swift as two words).

What's even worse is that most of my documents are in Chinese. Most Chinese words consist of
multiple characters. If you break them up, you end up with searching nonsense ("上海"=Shanghai, "上"=up,"海"=ocean).

I build this LLM tool to help me query my Notion documents with natural language.
"""
)

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
        # response = st.write_stream(response_generator())
        response = st.write("hey")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
