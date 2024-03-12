import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Implementation",
    page_icon="⚙️",
)
st.markdown("## What's under the hood? ⚙️")

st.markdown(
    """
My Notion Companion is a LLM-powered conversational RAG to chat with documents from Notion. It uses only **open-sourced technologies** and can **run on a single Mac Mini**.

Empowering technologies:
- **The Framework**: uses [Langchain](https://python.langchain.com/docs/)
- **The LLM**: uses 🤗-developed [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta). It has great inference speed, bilingual and instruction following capabilities
- **The Datastores**: the documents were stored into both conventional lexical data form and embeeding-based vectorstore (uses [Redis](https://python.langchain.com/docs/integrations/vectorstores/redis))
- **The Embedding Model**: uses [`sentence-transformers/distiluse-base-multilingual-cased-v1`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1). It has great inference speed and bilingual capability
- **The Tokenizers**: uses 🤗's [`AutoTokenizer`](AutoTokenizer) and Chinese text segmentation tool [`jieba`](https://github.com/fxsjy/jieba) (only in lexical search)
- **The Lexical Search Tool**: uses [`rank_bm25`](https://github.com/dorianbrown/rank_bm25)
- **The Computing**: uses [LlamaCpp](https://github.com/ggerganov/llama.cpp) to power the LLM in the local machine (a Mac Mini with M2 Pro chip)
- **The Observability Tool**: uses [LangSmith](https://docs.smith.langchain.com/)
- **The UI**: uses [Streamlit](https://docs.streamlit.io/)
"""
)


st.markdown(
    """
#### The E2E Pipeline

- When a user enters a prompt, the assistant will try lexical search first
  - a query analyzer will analyze the query and extract keywords (for search) and domains (for metadata filtering)
  - the extracted domains will be compared against the metadata of documents, only those with a matched metadata will be retrieved
  - the keyword will be segmented into searchable tokens, then further compared against the metadata-filtered documents with BM25 lexical search algorithm
  - The fetched documents will be subject to a final match checker to ensure relevance
- If lexical search doesn't return enough documents, the assistant will then try semantic search into the Redis vectorstore. Retrieved docs will also subject the QA by match checker.
- All retrieved documents will be sent to LLM as part of a system prompt, the LLM will then act as a conversational RAG to chat with the user with knowledges from the provided documents

"""
)

st.image("resources/flowchart.png", caption="E2E workflow")


st.markdown(
    """
#### Selecting the right LLM

I have compared a wide range of Bi/Multi-lingual LLMs with 7B parameters that has a LlamaCpp-friendly gguf executable on HuggingFace (which can fit onto Mac Mini's GPU).

I created conversational test cases to assess the models' instruction following, reasoning, helpfulness, coding, hallucinations and inference speed.
Qwen models (Qwen 1.0 & 1.5), together with HuggingFace's zephyr-7b-beta come as the top 3, but Qwen models are overly creative and do not follow few-shot examples.
Thus, the final candidate goes to **zephyr**.

Access the complete LLM evaluation results [here](https://docs.google.com/spreadsheets/d/1OZKu6m0fHPYkbf9SBV6UUOE_flgBG7gphgyo2rzOpsU/edit?usp=sharing).
"""
)

df_llm = pd.read_csv("resources/llm_scores.csv", index_col=0)

st.dataframe(df_llm)

st.markdown(
    """
#### Selecting the right LLM Computing Platform

I tested [Ollama](https://ollama.com/) first given its integrated, worry-free experiences that abstracted away the complexity of building environments and downloading LLMs.
However, I hit some unresponsiveness when experimenting with different LLMs and switched to [LlamaCpp](https://github.com/ggerganov/llama.cpp) (one layer deeper as the empowering backend for Ollama)

It works great so I sticked around.
"""
)

st.markdown(
    """
#### Selecting the right Vectordatabase

Langchain supports a huge number of vectordatabases. Because I don't have any scalability concerns (<300 docs in total),
I target on easiness, can run in local machine, supports to offload data into disk, and metadata fuzzy match.

Redis ended up being the only option that satisfies all the criteria.
"""
)

df_vs = pd.read_csv("resources/vectordatabase_evaluation.csv", index_col=0)

st.dataframe(df_vs)

st.markdown(
    """
#### Selecting the right Embedding model


"""
)

st.markdown(
    """
#### Selecting the right Observability Tool

Langchain ecosystem comes with its own [LangSmith](https://www.langchain.com/langsmith) observability tool. It works out of the box with minimal configurations and requires no change in codes.
"""
)


st.markdown(
    """
#### Selecting the right UI

[Streamlit](https://docs.streamlit.io/) and [Gradio](https://www.gradio.app/docs/) are among the popular options to share a LLM-based application.

I chose Streamlit for its script-writing development experience and integrated webapp-like UI that supports multi-page app creation.
"""
)

st.markdown(
    """
#### Appendix: Project Working Log and Feature Tracker

- [Working log](https://fredyang0507.notion.site/MyNotionCompanion-ce12513756784d2ab15015582538825e?pvs=4)
- [Feature Tracker](https://fredyang0507.notion.site/306e21cfd9fa49b68f7160b2f6692f72?v=789f8ef443f44c96b7cc5f0c99a3a773&pvs=4)
"""
)
