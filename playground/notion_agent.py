import json
import tomllib

from typing import Dict, Any, List, Sequence, Union, Tuple

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer
from transformers.pipelines.conversational import Conversation
from langchain_core.prompt_values import StringPromptValue

from langchain_community.vectorstores import Redis
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from langchain.chains.query_constructor.base import AttributeInfo

from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain.chains.query_constructor.prompt import USER_SPECIFIED_EXAMPLE_PROMPT, SUFFIX_WITHOUT_DATA_SOURCE
from langchain.retrievers.self_query.redis import RedisTranslator
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import StructuredQueryOutputParser
from langchain.output_parsers.boolean import BaseOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_core.documents.base import Document

class ChatBot:
    def __init__(self, llm: LlamaCpp, config: Dict[str, Any]):

        self.llm = llm
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'],
            trust_remote_code=True
        )

        self.model_params = config['llm']
        self.conversation = Conversation()
        self.full_history = Conversation()
        
        self.sys_msg = {
            "role": "system",
            "content": """You are a helpful assistant. You only answer questions you are very sure of. \
When you don't know, say "I don't know." Avoid not replying at all. Please answer questions in the language being asked.\
你是一个友好而乐于助人的AI助手。\
你只回答你非常确定的问题。如果你不知道，你会如实回答“我不知道。”不能拒绝回答问题。请使用提问使用的语言进行回答。""",
        }

        # add system message to the conversation history
        self.conversation.add_message(self.sys_msg)
        self.full_history.add_message(self.sys_msg)

        self.prompt = PromptTemplate.from_template("{message}")
        
        self.chain = self.prompt | self.llm

    def convert_message_to_llm_format(self, msg):
        # https://huggingface.co/docs/transformers/chat_templating
        return self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def invoke(self, text: str) -> Dict[str, str]:

        # convert StringPromptValue (str like object but with format check) to string
        # it is incomptible with tokenizer.apply_chat_template()
        # ref: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompt_values.py
        if isinstance(text, StringPromptValue):
            text = text.to_string()
        
        inputs = {
            "role": "user", 
            "content": text,
        }
        
        # add the message to the memeory
        self.conversation.add_message(inputs)
        self.full_history.add_message(inputs)
        
        inputs = {'message': self.convert_message_to_llm_format(self.conversation)}

        # invoke chain and format to Conversation-style response
        response = {
            "role": "assistant",
            "content": self.chain.invoke(inputs),
        }

        # add response to memory
        self.conversation.add_message(response)
        self.full_history.add_message(response)

        # prevent memory overflow
        self._keep_k_rounds_most_recent_conversation()
        
        return response

    # def __call__(self, text: str):
    #     # have to create a __call__ interface for SelfQueryRetriever constructor
    #     # otherwise hit TypeError: Expected a Runnable, callable or dict.Instead got an unsupported type: <class '__main__.chatbot'>
    #     self.invoke(text)
        
    def clear_conversation(self):
        self.conversation = Conversation()

    def _keep_k_rounds_most_recent_conversation(self):
        k = self.model_params['conversation']['k_rounds']
        if len(self.conversation) > 2*k:
            # keep if system input exists
            if self.conversation[0]['role'] == 'system':
                self.conversation = Conversation([self.conversation[0]] + self.conversation[-2*k:])
            else:
                self.conversation = Conversation(self.conversation[-2*k:])
                
    def extract_ai_responses(self):
        return self.full_history.generated_responses


class SelfQueryAgent:

    def __init__(self, llm: LlamaCpp, config: Dict[str, Any], token: Dict[str, str], enable_compressor: bool = True) -> None:
        
        # init embedding model
        self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key=token['huggingface'], 
            model_name=config['embedding_model']
        )

        # init redis vectorstore
        self.vs = Redis.from_existing_index(
            embedding=self.embedding_model,
            index_name=config['index_name'],
            redis_url=config['redis_url'],
            schema=config['redis_schema'],
        )

        self.llm = llm
        self.config = config

        self._check_schema_definition()
        
        self.retriever = self.construct_retriever()
        if self.config['enable_compressor']:
            self.compression_retriever = self.construct_compression_retriever()
        

    def _check_schema_definition(self) -> None:
        metadata_vectorstore = set(
            [x['name'] for x in self.vs.schema['text']] + [x['name'] for x in self.vs.schema['numeric']]
        )
        # ensure there's no more undocumented metadata
        # redis will auto-generate a "content" attribute
        assert metadata_vectorstore == set(self.config['attributes'].keys()).union(['content'])


    def get_attributes_info(self) -> List[AttributeInfo]:
        attribute_info = list()
        for k, v in self.config['attributes'].items():
            attribute_info.append(
                AttributeInfo(
                    name=k,
                    description=v['description'],
                    type=v['type']
                )
            )
        return attribute_info
    

    def _get_self_query_prompt(self, template, examples) -> FewShotPromptTemplate:

        examples = self._construct_examples(
            [(x['user_query'], x['structured_request']) for x in examples['example']]
        )

        self_query_prompt = FewShotPromptTemplate(
            examples=list(examples),
            example_prompt=USER_SPECIFIED_EXAMPLE_PROMPT,
            input_variables=["query"],
            suffix=SUFFIX_WITHOUT_DATA_SOURCE.format(i=len(examples) + 1),
            prefix=template.format(
                content_and_attributes=json.dumps({
                    'content': self.config['content'],
                    'attributes': self._format_attribute_info(self.get_attributes_info())
                }, indent=4, ensure_ascii=False).replace("{", "{{").replace("}", "}}"),
                attributes_set=str(list(self.config['attributes'].keys()))
            )
        )

        return self_query_prompt
    
    def construct_retriever(self):

        with open(self.config['self_query']['examples'], 'rb') as f:
            self_query_examples = tomllib.load(f)

        with open(self.config['self_query']['template'], 'r') as f:
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
    def _construct_examples(input_output_pairs: Sequence[Tuple[str, dict]]) -> List[dict]:
        """Construct examples from input-output pairs.

        Adapted from: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/query_constructor/base.py
        """
        examples = []
        for i, (_input, output) in enumerate(input_output_pairs):
            structured_request = (
                json.dumps(output, indent=4, ensure_ascii=False).replace("{", "{{").replace("}", "}}")
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

        with open(self.config['compressor']['template'], 'r') as f:
            compressor_template = "\n".join(f.readlines())
    
        compressor = LLMChainFilter.from_llm(
            self.llm,
            prompt=self._get_compressor_prompt(compressor_template)
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=self.retriever,
        )

        return compression_retriever
    
    def invoke(self, query: str) -> List[Document]:
        if self.config['enable_compressor']:
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
            
        cleaned_text = text.split("\n")[0].split(" ")[0].split("。")[0] # only extract the first word
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