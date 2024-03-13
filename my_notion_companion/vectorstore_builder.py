from typing import Any, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Redis
from loguru import logger

from my_notion_companion.utils import load_notion_documents


class RedisIndexBuilder:
    """Build Redis index."""

    def __init__(
        self,
        config: Dict[str, Any],
        tokens: Dict[str, str],
    ):

        self.config = config
        self.tokens = tokens
        self.docs = load_notion_documents(self.config, self.tokens)

        self._init_embedding_model()
        self._split_documents()
        self._check_schema_match()

    def _init_embedding_model(self) -> None:
        self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key=self.tokens["huggingface"],
            model_name=self.config["embedding_model"],
        )

    def _split_documents(self) -> None:
        # Presumably docs in NotionDB fits more with MarkdownHeaderTextSplitter
        # however, most of the documents in my personal databases don't have such header-text structure
        # and they are not important for my use cases (I won't ask it to reason on a specific section
        # in a doc). Thus I'll use the regular RecursiveCharacterTextSplitter
        rc_splitter = RecursiveCharacterTextSplitter(**self.config["splitter"])
        self.splits = rc_splitter.split_documents(self.docs)

    def _check_schema_match(self) -> None:
        # metadata from notion document
        metadata_set = set()
        for metadata in [x.metadata for x in self.docs]:
            metadata_set = metadata_set.union(list(metadata.keys()))

        # metadata definition from config file
        metadata_config = set(
            [x["name"] for x in self.config["redis_schema"]["text"]]
            + [x["name"] for x in self.config["redis_schema"]["numeric"]]
        )
        # make sure we defined the schema for all metadata in _CONFIG file
        assert metadata_set == metadata_config

    def build_index(self) -> None:

        # drop index if exists
        Redis.drop_index(
            redis_url=self.config["redis_url"],
            index_name=self.config["index_name"],
            delete_documents=True,
        )
        # Redis supports default "tag" fields alongside with "text" and "numeric"
        # looks like a better match for "tags" property at the first glance
        # but we'll classify it as "text" anyway because the to give consistency of
        # how downstream self-query writes filter queries.
        # ref: https://redis.io/docs/interact/search-and-query/advanced-concepts/tags
        Redis.from_documents(
            documents=self.splits,
            embedding=self.embedding_model,
            redis_url=self.config["redis_url"],
            index_name=self.config["index_name"],
            index_schema=self.config["redis_schema"],
        )

        # can use `!rvl index info -i notiondb` to peek the information
        logger.info(f"Index built in redis: {self.config['index_name']}")
