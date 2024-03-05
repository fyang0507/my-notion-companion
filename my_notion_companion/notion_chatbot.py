import os
import pickle
import tomllib
from typing import Any, Dict, List

from conversational_rag import ConversationalRAG
from langchain_community.llms import LlamaCpp
from langchain_core.documents.base import Document
from loguru import logger
from notion_loader import NotionLoader
from retriever import BM25SelfQueryRetriever
from utils import peek_docs


class NotionChatBot:
    def __init__(self, llm: LlamaCpp, config_path: str) -> None:

        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        with open(self.config["template"]["conversatoinal_rag"], "rb") as f:
            self.system_message = tomllib.load(f)["system"]

        if not self.config["force_repull"] and os.path.exists(
            self.config["path"]["docs"]
        ):
            with open(self.config["path"]["docs"], "rb") as f:
                docs = pickle.load(f)
        else:
            with open(self.config["path"]["notion_dbs"], "rb") as f:
                _DATABASES_NOTION = tomllib.load(f)

            with open(self.config["path"]["tokens"], "rb") as f:
                _TOKENS = tomllib.load(f)

            loader = NotionLoader(_TOKENS, _DATABASES_NOTION)
            loader.export_to_pickle(self.config["path"]["docs"])
            docs = loader.load()

        # create llm
        # self.llm = LlamaCpp(
        #     model_path=self.config['model_path']+'/'+self.config['model_mapping'][self.config['model_name']],
        #     name=self.config['model_name'],
        #     **self.config['llm']
        # )
        self.llm = llm

        self.retriever = BM25SelfQueryRetriever(self.llm, docs, self.config)
        self.n_query = 0

    def invoke(self, query: str) -> str:
        self.n_query += 1

        if self.n_query == 1:
            docs_retrieved: List[Document] = self.retriever.invoke(query)
            logger.info(peek_docs(docs_retrieved))
            self.conversatoinal_rag = ConversationalRAG(
                self.llm, self.config, self.system_message, docs_retrieved
            )

        return self.conversatoinal_rag.invoke(query)
