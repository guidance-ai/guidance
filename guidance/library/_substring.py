import re
from typing import Callable, Iterable, Literal, Optional, Union

from .._ast import RuleNode, SubstringNode


def chunk_on_word(text: str) -> list[str]:
    return re.findall(r"(\s+|\w+|[^\s\w]+)", text)


def substring(
    target_string: str,
    *,
    chunk: Union[Literal["word", "character"], Callable[[str], Iterable[str]]] = "word",
    name: Optional[str] = None,
) -> RuleNode:
    if chunk == "word":
        chunks = chunk_on_word(target_string)
    elif chunk == "character":
        chunks = tuple(target_string)
    elif callable(chunk):
        chunks = chunk(target_string)
        if "".join(chunks) != target_string:
            raise ValueError(
                "chunk_on function must return a sequence of strings that can be joined to form the target string"
            )
    else:
        raise ValueError(f"Invalid `chunk` value: {chunk!r}. Expected 'word', 'character', or a function.")

    return RuleNode(
        name=name or "substring",
        value=SubstringNode(tuple(chunks)),
        capture=name,
    )
