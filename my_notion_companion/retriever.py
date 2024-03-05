import json
import re
import time
import tomllib
from typing import Any, Dict, List, Sequence, Tuple, Union

import jieba
from langchain.chains.query_constructor.base import (
    AttributeInfo,
    StructuredQueryOutputParser,
)
from langchain.chains.query_constructor.prompt import (
    SUFFIX_WITHOUT_DATA_SOURCE,
    USER_SPECIFIED_EXAMPLE_PROMPT,
)
from langchain.output_parsers.boolean import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers import (
    BM25Retriever,
    ContextualCompressionRetriever,
    SelfQueryRetriever,
)
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.self_query.redis import RedisTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Redis
from langchain_core.documents.base import Document
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from loguru import logger
from utils import load_test_cases

from my_notion_companion.document_filter import DocumentFilter
from my_notion_companion.query_analyzer import QueryAnalyzer


class BasicRetriever:
    def __init__(
        self,
        config: Dict[str, Any],
        token: Dict[str, str],
    ) -> None:

        # init embedding model
        self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key=token["huggingface"], model_name=config["embedding_model"]
        )
        self.vs = Redis.from_existing_index(
            embedding=self.embedding_model,
            index_name=config["index_name"],
            redis_url=config["redis_url"],
            schema=config["redis_schema"],
        )
        self.retriever = self.vs.as_retriever()

    def invoke(self, query):
        return self.retriever.invoke(query)


class BM25SelfQueryRetriever:
    def __init__(
        self, llm: LlamaCpp, docs: List[Document], config: Dict[str, Any]
    ) -> None:

        self.llm = llm
        self.config = config
        self.docs = docs

        self._split_documents()

        self.doc_filter = DocumentFilter(self.splits, threshold=0.8)
        self.query_analyzer = QueryAnalyzer(llm, config, verbose=True)

    def _split_documents(self) -> None:
        rc_splitter = RecursiveCharacterTextSplitter(**self.config["splitter"])
        self.splits = rc_splitter.split_documents(self.docs)

    def _get_relevant_documents(
        self, query: str, splits: List[Document]
    ) -> List[Document]:
        retriever = BM25Retriever.from_documents(
            splits,
            k=self.config["bm25"]["k"],
            preprocess_func=lambda x: jieba.lcut_for_search(x, HMM=False),
        )
        return retriever.invoke(query)

    def _filter_documents(self, query_formatted: Dict[str, str]) -> List[Document]:
        keywords: List[str] = query_formatted.get("keywords")
        domains: List[str] = query_formatted.get("domains")

        if len(domains) == 1 and domains[0] == "无":
            logger.info("No filters found by query analyzer.")
            splits = self.splits
        else:
            logger.info(f"filter found by query analyzer: {domains}")
            try:
                splits = self.doc_filter.filter_multiple_criteria(domains, operand="OR")
            except RuntimeError:
                logger.info(
                    "No matched doc found based on query analyzer. Search all docs."
                )
                splits = self.splits

        splits_matched = list()
        for keyword in keywords:
            splits_matched.extend(self._get_relevant_documents(keyword, splits))

        return splits_matched

    def invoke(self, query: str) -> List[Document]:
        try:
            query_formatted = self.query_analyzer.invoke(query)
        except RuntimeError:
            logger.error(
                f"Failed to construct query for the input: {query}, returning raw input."
            )
            return self._get_relevant_documents(query, self.splits)

        return self._filter_documents(query_formatted)


class RedisSelfQueryRetriever:

    def __init__(
        self,
        llm: LlamaCpp,
        config: Dict[str, Any],
        token: Dict[str, str],
    ) -> None:

        # init embedding model
        self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key=token["huggingface"], model_name=config["embedding_model"]
        )

        # init redis vectorstore
        self.vs = Redis.from_existing_index(
            embedding=self.embedding_model,
            index_name=config["index_name"],
            redis_url=config["redis_url"],
            schema=config["redis_schema"],
        )

        self.llm = llm
        self.config = config
        self.enable_compressor = self.config["enable_compressor"]

        self._check_schema_definition()

        self.retriever = self.construct_retriever()

        if self.enable_compressor:
            self.compression_retriever = self.construct_compression_retriever()

    def _check_schema_definition(self) -> None:
        metadata_vectorstore = set(
            [x["name"] for x in self.vs.schema["text"]]
            + [x["name"] for x in self.vs.schema["numeric"]]
        )
        # ensure there's no more undocumented metadata
        # redis will auto-generate a "content" attribute
        assert metadata_vectorstore == set(self.config["attributes"].keys()).union(
            ["content"]
        )

    def get_attributes_info(self) -> List[AttributeInfo]:
        attribute_info = list()
        for k, v in self.config["attributes"].items():
            attribute_info.append(
                AttributeInfo(name=k, description=v["description"], type=v["type"])
            )
        return attribute_info

    def _get_self_query_prompt(self, template, examples) -> FewShotPromptTemplate:

        examples = self._construct_examples(
            [(x["user_query"], x["structured_request"]) for x in examples["example"]]
        )

        self_query_prompt = FewShotPromptTemplate(
            examples=list(examples),
            example_prompt=USER_SPECIFIED_EXAMPLE_PROMPT,
            input_variables=["query"],
            suffix=SUFFIX_WITHOUT_DATA_SOURCE.format(i=len(examples) + 1),
            prefix=template.format(
                content_and_attributes=json.dumps(
                    {
                        "content": self.config["content"],
                        "attributes": self._format_attribute_info(
                            self.get_attributes_info()
                        ),
                    },
                    indent=4,
                    ensure_ascii=False,
                )
                .replace("{", "{{")
                .replace("}", "}}"),
                attributes_set=str(list(self.config["attributes"].keys())),
            ),
        )

        return self_query_prompt

    def construct_retriever(self):

        with open(self.config["example"]["self_query"], "rb") as f:
            self_query_examples = tomllib.load(f)

        with open(self.config["template"]["self_query"], "r") as f:
            self_query_template = "\n".join(f.readlines())

        prompt = self._get_self_query_prompt(self_query_template, self_query_examples)
        output_parser = StructuredQueryOutputParser.from_components()

        retriever = SelfQueryRetriever(
            query_constructor=prompt | self.llm | output_parser,
            vectorstore=self.vs,
            # requires a RedisModel object as input
            # can use the private ._schema attribute
            # ref: https://github.com/langchain-ai/langchain/blob/d7c26c89b2d4f5ff676ba7c3ad4f9075d50a8ab7/libs/community/langchain_community/vectorstores/redis/base.py#L572
            structured_query_translator=RedisTranslator(schema=self.vs._schema),
            enable_limit=True,
        )
        return retriever

    @staticmethod
    def _format_attribute_info(info: Sequence[Union[AttributeInfo, dict]]) -> str:
        """Construct examples from input-output pairs.

        Adapted from: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/query_constructor/base.py
        """
        info_dicts = {}
        for i in info:
            i_dict = dict(i)
            info_dicts[i_dict.pop("name")] = i_dict
        return info_dicts

    @staticmethod
    def _construct_examples(
        input_output_pairs: Sequence[Tuple[str, dict]]
    ) -> List[dict]:
        """Construct examples from input-output pairs.

        Adapted from: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/query_constructor/base.py
        """
        examples = []
        for i, (_input, output) in enumerate(input_output_pairs):
            structured_request = (
                json.dumps(output, indent=4, ensure_ascii=False)
                .replace("{", "{{")
                .replace("}", "}}")
            )
            example = {
                "i": i + 1,
                "user_query": _input,
                "structured_request": structured_request,
            }
            examples.append(example)
        return examples

    def _get_compressor_prompt(self, template) -> PromptTemplate:
        return PromptTemplate(
            template=template,
            input_variables=["question", "context"],
            output_parser=ChineseBooleanOutputParser(),
        )

    def construct_compression_retriever(self):

        with open(self.config["template"]["compressor"], "r") as f:
            compressor_template = "\n".join(f.readlines())

        compressor = LLMChainFilter.from_llm(
            self.llm, prompt=self._get_compressor_prompt(compressor_template)
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever,
        )

        return compression_retriever

    def invoke(self, query: str) -> List[Document]:
        if self.enable_compressor:
            return self.compression_retriever.invoke(query)
        else:
            return self.retriever.invoke(query)


class ChineseBooleanOutputParser(BaseOutputParser[bool]):
    """Parse the output of an LLM call to a boolean.

    Adapted from: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/output_parsers/boolean.py
    """

    true_val: str = "是"
    """The string value that should be parsed as True."""
    false_val: str = "否"
    """The string value that should be parsed as False."""

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call to a boolean.

        Args:
            text: output of a language model

        Returns:
            boolean
        """
        # if llm.name == 'Qwen/Qwen-7B-Chat':
        #     text = text.replace("[PAD151645]", "\n")

        cleaned_text = (
            text.split("\n")[0].split(" ")[0].split("。")[0]
        )  # only extract the first word
        if cleaned_text.upper() not in (self.true_val.upper(), self.false_val.upper()):
            raise ValueError(
                f"BooleanOutputParser expected output value to either be "
                f"{self.true_val} or {self.false_val}. Received {cleaned_text}."
            )
        return cleaned_text.upper() == self.true_val.upper()

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "customized_boolean_output_parser"


class RetrieverEvaluator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.test_cases = load_test_cases(config["test_path"])

    def evaluate(self, retriever):
        score_list = list()

        for case in self.test_cases:
            time.sleep(0.1)
            docs_retrieved = retriever.invoke(case["q"])
            for ref in case["docs"]:
                score_list.append(self._match_chinese(ref, docs_retrieved))

        logger.info(f"Test cases pass rate: {sum(score_list)/len(score_list):.3f}")
        return score_list

    @staticmethod
    def _match_chinese(string_to_match: str, docs: List[Document]):
        """Check whether Chinese characters in string_to_match appear in at least one doc."""
        # find all chinese characters
        # ref: https://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex
        for tokens in re.findall(r"[\u4e00-\u9fff]+", string_to_match):
            return any([tokens in x.page_content for x in docs])
