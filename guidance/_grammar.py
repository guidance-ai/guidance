import dataclasses
import re
from typing import Optional, Sequence, Union

from ._ast import (
    Function,
    GrammarNode,
    LiteralNode,
    RegexNode,
    RepeatNode,
    RuleNode,
    SelectNode,
    SubgrammarNode,
    _parse_tags,
)

def string(s: str) -> LiteralNode:
    return LiteralNode(s)


def regex(pattern: str) -> RegexNode:
    return RegexNode(pattern)


def gen(
    regex: Optional[str] = None,
    stop: Optional[str] = None,
    stop_regex: Optional[str] = None,
    suffix: Optional[str] = None,
    stop_capture: Optional[str] = None,
    name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    list_append: bool = False,
) -> RuleNode:
    if stop is not None and stop_regex is not None:
        raise ValueError("You cannot specify both a stop and a stop_regex")
    if stop is not None:
        stop_value = LiteralNode(stop)
    elif stop_regex is not None:
        stop_value = RegexNode(stop_regex)
    else:
        stop_value = None

    node = RuleNode(
        name=name or "gen",
        value=RegexNode(regex),
        capture=name,
        stop=stop_value,
        suffix=LiteralNode(suffix) if suffix else None,
        stop_capture=stop_capture,
        list_append=list_append,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return node


def select(
    options: Sequence[Union[str, int, float, GrammarNode]],
    name: Optional[str] = None,
    list_append: bool = False,
) -> GrammarNode:
    """Choose between a set of options.

    This function constrains the next generation from the LLM to be one of the
    given `options`.
    If the list only has a single element, then that value can be returned
    immediately, without calling the LLM.

        >>> lm += select(["Temeraire", "Redoutable", "Bucentaure"], name="my_selection")
        >>> print(lm["my_selection"])
        Temeraire

    Parameters
    ----------
    name : str or None
        If this is not None then the the results of the generation will be saved as a variable on
        the Model object (so you can access the result as `lm["var_name"]`).

    options : list
        The set of available choices for the next generation

    list_append : bool
        If this is True then the results saved to `lm[name]` will not be written directly but rather appended
        to a list (if no list with the current name is present one will be created). This is useful for
        building lists inside python loops.
    """
    alternatives: list[GrammarNode] = []
    for v in options:
        if isinstance(v, (int, float)):
            alternatives.append(string(str(v)))
        elif isinstance(v, str):
            node = _parse_tags(v)
            if isinstance(node, Function):
                raise ValueError(
                    "You cannot select between stateful functions in the current guidance implementation!"
                )
            if callable(node):
                raise ValueError(
                    "Did you pass a function without calling it to select? You need to pass the results of a called guidance function to select."
                )
            alternatives.append(node)
        elif isinstance(v, GrammarNode):
            alternatives.append(v)
        else:
            raise ValueError(f"Option {v} is not a valid type: {type(v)}")

    return RuleNode(
        name=name or "select",
        value=SelectNode(tuple(alternatives)),
        capture=name,
        list_append=list_append,
    )


def repeat(
    value: Union[str, int, float, GrammarNode], min: int, max: Optional[int] = None
) -> GrammarNode:
    if isinstance(value, (int, float)):
        node = string(str(value))
    elif isinstance(value, str):
        node = _parse_tags(value)
        if isinstance(node, Function):
            raise ValueError(
                "You cannot repeat a stateful function in the current guidance implementation!"
            )
        if callable(node):
            raise ValueError(
                "Did you pass a function without calling it? You need to pass the results of a called guidance function to repeat."
            )
    elif isinstance(value, GrammarNode):
        node = value
    else:
        raise ValueError(f"Value {value} is not a valid type: {type(value)}")

    return RuleNode(
        name="repeat",
        value=RepeatNode(node, min, max),
    )


def token_limit(value: GrammarNode, max_tokens: int) -> RuleNode:
    """This sets the token limit to be used for the given portion of the grammar."""
    try:
        if isinstance(value, RuleNode):
            return dataclasses.replace(value, max_tokens=max_tokens)
        else:
            return RuleNode(name="token_limit", value=value, max_tokens=max_tokens)
    except ValueError:
        return RuleNode(name="token_limit", value=subgrammar(value), max_tokens=max_tokens)


def with_temperature(value: GrammarNode, temperature: float) -> RuleNode:
    """This sets the sampling temperature to be used for the given portion of the grammar.

    Note that if the grammar passed to us already has some portions with a temperature
    setting in place, those settings will not be overridden.
    """
    try:
        if isinstance(value, RuleNode):
            return dataclasses.replace(value, temperature=temperature)
        else:
            return RuleNode(name="with_temperature", value=value, temperature=temperature)
    except ValueError:
        return RuleNode(name="with_temperature", value=subgrammar(value), temperature=temperature)


def capture(value: GrammarNode, name: str, list_append: bool = False) -> RuleNode:
    if isinstance(value, RuleNode) and value.capture is None:
        return dataclasses.replace(value, capture=name, list_append=list_append)
    else:
        return RuleNode(name="capture", value=value, capture=name, list_append=list_append)


def subgrammar(body: GrammarNode, name: Optional[str] = None, skip_regex: Optional[str] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> SubgrammarNode:
    capture_name = name
    name = name or (body.name if isinstance(body, RuleNode) else "subgrammar")
    node = SubgrammarNode(name=name, body=body, skip_regex=skip_regex)
    if max_tokens:
        node = token_limit(node, max_tokens)
    if temperature:
        node = with_temperature(node, temperature)
    if capture_name:
        node = capture(node, capture_name)
    return node


def quote_regex(value: str) -> str:
    return re.sub(r"([\\+*?^$(){}\[\]\.|])", r"\\\1", value)
