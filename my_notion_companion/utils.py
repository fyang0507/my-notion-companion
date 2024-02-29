import re
from typing import List, Dict


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
