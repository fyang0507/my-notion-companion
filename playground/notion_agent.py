from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer
from transformers.pipelines.conversational import Conversation
from langchain_core.prompt_values import StringPromptValue

class chatbot:
    def __init__(self, model_name, model_path, **model_params):

        self.llm = LlamaCpp(model_path=model_path, name=model_name, **model_params)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.model_params = dict(model_params)
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

    def invoke(self, text: str):

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

    def __call__(self, text: str):
        # have to create a __call__ interface for SelfQueryRetriever constructor
        # otherwise hit TypeError: Expected a Runnable, callable or dict.Instead got an unsupported type: <class '__main__.chatbot'>
        self.invoke(text)
        
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