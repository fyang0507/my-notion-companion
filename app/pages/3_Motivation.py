"""
Ref: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
"""

import streamlit as st

st.set_page_config(
    page_title="Motivation",
    page_icon="🤨",
)

st.markdown("## So Fred, why did you start this project? 🤨")

st.markdown(
    """
As much as I've been a very loyal (but freemium) Notion user, search func in Notion **sucks**. It supports only discrete keyword search with exact match (e.g. it treats Taylor Swift as two words).

What's even worse is that most of my documents are in Chinese. Most Chinese words consist of
multiple characters. If you break them up, you end up with a total different meaning ("上海"=Shanghai, "上"=up,"海"=ocean).
"""
)

st.image(
    "resources/search-limit-chinese.png",
    caption="tried to search for 天马 Pegasus, but it ends up with searching two discrete characters 天 sky and 马 horse",
)

st.markdown(
    """
My Notion Compnion is here to help me achieve two things:
- to have an improved search experience across my notion databases (200+ documents)
- to chat with my Notion documents in natural language
"""
)
