import re
from typing import List


_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_./@+-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """Tokenize mixed Chinese/English text without external dependencies.

    Chinese runs emit characters and adjacent bigrams. The tokenizer is a
    deterministic baseline, not a replacement for a production segmenter.
    """

    tokens: List[str] = []
    for match in _TOKEN_PATTERN.finditer(text.lower()):
        value = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", value):
            chars = list(value)
            tokens.extend(chars)
            tokens.extend(chars[index] + chars[index + 1] for index in range(len(chars) - 1))
        else:
            tokens.append(value)
    return tokens

