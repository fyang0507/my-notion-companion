import re


def fix_qwen_padding(s: str) -> str:
    """Fix qwen model incompatible EOS tokens.

    Qwen has some weird compatibility issue with LlamaCpp such that it will generate EOS
    tokens like [PAD151645], [PAD151643], etc.
    ref: https://github.com/ggerganov/llama.cpp/issues/4331
    """
    return re.sub(r"\[PAD[0-9]+\]([\s\w\W\S]*)", "", s)  # trim anything beyond [PADxx]
