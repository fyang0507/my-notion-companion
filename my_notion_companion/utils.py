import re
from typing import Dict, List

from langchain_core.documents.base import Document
from transformers import AutoTokenizer
from transformers.pipelines.conversational import Conversation


def fix_qwen_padding(s: str) -> str:
    """Fix qwen model incompatible EOS tokens.

    Qwen has some weird compatibility issue with LlamaCpp such that it will generate EOS
    tokens like [PAD151645], [PAD151643], etc.
    ref: https://github.com/ggerganov/llama.cpp/issues/4331
    """
    return re.sub(
        r"\[PAD[0-9]+\]([\s\w\W\S]*)", "", s
    )  # trim anything beyond the EOS indicator [PADxx]


def load_test_cases(path: str) -> List[Dict[str, str]]:
    with open("../data/test_cases.txt") as f:
        raw = f.readlines()

    raw = "".join(raw[0::2]).split("问：")[1:]
    raw = [re.split(r"\n答：|\n资料：", x) for x in raw]
    test_cases = [{"q": x[0], "a": x[1], "docs": x[2:]} for x in raw]

    return test_cases


def format_docs(docs: List[Document]) -> str:
    def format_doc_with_metadata(doc: Document) -> str:
        return "内容：\n" + doc.page_content + "\n\n元数据：\n" + str(doc.metadata)

    s = ""
    for idx, doc in enumerate(docs):
        s += f"文档{idx+1}\n\n"
        s += format_doc_with_metadata(doc)
        s += "\n\n\n"

    return s[:-3]  # skip the last \n\n\n


def convert_message_to_llm_format(tokenizer: AutoTokenizer, msg: Conversation) -> str:
    # https://huggingface.co/docs/transformers/chat_templating
    return tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )


def peek_docs(docs: List[Document]) -> str:
    result = ""
    for doc in docs:
        result += str(doc.metadata)
        result += "\n\n"
        result += doc.page_content[:30]
        result += "...\n"
        result += "-" * 30
        result += "\n"

    return result
