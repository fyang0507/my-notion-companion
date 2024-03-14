# My Notion Companion 🤖
###### A conversational RAG that helps to chat with my (mostly Chinese-based) Notion Databases.

My Notion Companion is a LLM-powered conversational RAG to chat with documents from Notion.
It uses hybrid search (lexical + semantic) search to find the relevant documents and a chat interface to interact with the docs.
It uses only **open-sourced technologies** and can **run on a single Mac Mini**.


### Empowering technologies
- **The Framework**: uses [Langchain](https://python.langchain.com/docs/)
- **The LLM**: uses 🤗-developed [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta). It has great inference speed, bilingual and instruction following capabilities
- **The Datastores**: the documents were stored into both conventional lexical data form and embeeding-based vectorstore (uses [Redis](https://python.langchain.com/docs/integrations/vectorstores/redis))
- **The Embedding Model**: uses [`sentence-transformers/distiluse-base-multilingual-cased-v1`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1). It has great inference speed and bilingual capability
- **The Tokenizers**: uses 🤗's [`AutoTokenizer`](AutoTokenizer) and Chinese text segmentation tool [`jieba`](https://github.com/fxsjy/jieba) (only in lexical search)
- **The Lexical Search Tool**: uses [`rank_bm25`](https://github.com/dorianbrown/rank_bm25)
- **The Computing**: uses [LlamaCpp](https://github.com/ggerganov/llama.cpp) to power the LLM in the local machine (a Mac Mini with M2 Pro chip)
- **The Observability Tool**: uses [LangSmith](https://docs.smith.langchain.com/)
- **The UI**: uses [Streamlit](https://docs.streamlit.io/)

### What's wrong with Notion's native search?
As much as I've been a very loyal (but freemium) Notion user, search func in Notion **sucks**. It supports only discrete keyword search with exact match (e.g. it treats Taylor Swift as two words).

What's even worse is that most of my documents are in Chinese. Most Chinese words consist of multiple characters. If you break them up, you end up with a total different meaning ("上海"=Shanghai, "上"=up,"海"=ocean).

My Notion Compnion is here to help me achieve two things:
- to have an improved search experience across my notion databases (200+ documents)
- to have a conversation with my Notion documents in natural language

#### The system design
![e2e_pipeline](resources/flowchart.png)

See more detailed evaluation and documentation on [Fred's 🤗 space](https://huggingface.co/spaces/fyang0507/my-notion-companion)!

### Quickstart



### References
- [Working Log](https://fredyang0507.notion.site/MyNotionCompanion-ce12513756784d2ab15015582538825e?pvs=4)
- [Feature Tracker](https://fredyang0507.notion.site/306e21cfd9fa49b68f7160b2f6692f72?v=789f8ef443f44c96b7cc5f0c99a3a773&pvs=4)
