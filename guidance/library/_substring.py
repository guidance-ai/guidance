import re
from typing import Callable, Iterable, Literal, Optional, Union

from .._ast import RuleNode, SubstringNode


def chunk_on_word(text: str) -> list[str]:
    return re.findall(r"(\s+|\w+|[^\s\w]+)", text)


def substring(
    target_string: str,
    *,
    chunk_on: Union[Literal["word", "character"], Callable[[str], Iterable[str]]] = "word",
    name: Optional[str] = None,
) -> RuleNode:
    if chunk_on == "word":
        chunks = chunk_on_word(target_string)
    elif chunk_on == "character":
        chunks = list(target_string)
    elif callable(chunk_on):
        chunks = chunk_on(target_string)
        if "".join(chunks) != target_string:
            raise ValueError(
                "chunk_on function must return a sequence of strings that can be joined to form the target string"
            )
    else:
        raise ValueError(f"Invalid chunk_on value: {chunk_on}. Expected 'word', 'character', or a function.")

    return RuleNode(
        name=name or "substring",
        value=SubstringNode(chunks),
        capture=name,
    )
