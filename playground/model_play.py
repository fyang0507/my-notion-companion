from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import json

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

llm = LlamaCpp(
    model_path='/Users/fred/Documents/models/chinese-alpaca-2-7b-q4_0.gguf',
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,
    verbose=True,
)


# prompt template from example script
# https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/langchain/langchain_qa.py
# official guide from HF: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
prompt_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。\n"
    "<</SYS>>\n\n"
    "{context}\n{question} [/INST]"
)

prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

chain = prompt | llm
chain.invoke({'context': '', 'question': '你是谁？'})

llm.invoke("who are you?")
llm.invoke("你是谁？")
llm.invoke("谁创造了你？")

llm.invoke("介绍一下孔子，200字以内。")

# mv ~/Downloads/*.gguf ~/Documents/models/

# available models:

# yi-chat-6b.Q4_K_M.gguf
# Qwen-7B-Chat.Q4_K_M.gguf
# baichuan2-7b-chat.Q4_K_S.gguf
# zephyr-7b-beta.Q4_K_M.gguf
# chinese-alpaca-2-7b-q4_0.gguf # don't fucking use it, trash
