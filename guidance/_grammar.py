import copy
import dataclasses
import re
from typing import Optional, Sequence, Union

from ._ast import (
    Function,
    GenNode,
    GrammarNode,
    LiteralNode,
    RegexNode,
    RepeatNode,
    RuleNode,
    SelectNode,
    SubgrammarNode,
    parse_tags,
)

# TODO: maybe regex, gen, select, and repeat defined here should be private api with public wrappers in library?


def string(s: str) -> LiteralNode:
    return LiteralNode(s)


def regex(pattern: str) -> RegexNode:
    return RegexNode(pattern)


def gen(
    regex: str = "(?s).*",
    stop_regex: str = "",
    save_stop_text: Optional[str] = None,
    name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> GenNode:
    node = GenNode(
        name=name or "gen",
        value=RegexNode(regex),
        capture=name,
        stop_regex=stop_regex,
        save_stop_text=save_stop_text,
    )
    node.temperature = temperature
    node.max_tokens = max_tokens
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
            node = parse_tags(v)
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

    capture_name: Optional[str] = name
    if list_append:
        if name is None:
            raise ValueError("list_append requires a name")
        capture_name = f"__LIST_APPEND:{name}"

    return RuleNode(
        name=name or "select",
        value=SelectNode(alternatives),
        capture=capture_name,
    )


def repeat(
    value: Union[str, int, float, GrammarNode], min: int, max: Optional[int] = None
) -> GrammarNode:
    if isinstance(value, (int, float)):
        node = string(str(value))
    elif isinstance(value, str):
        node = parse_tags(value)
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
    if isinstance(value, RuleNode):
        rule = copy.copy(value)
    else:
        try:
            rule = RuleNode(name="token_limit", value=value)
            rule.max_tokens = max_tokens
        except ValueError:
            inner_name = value.name if isinstance(value, RuleNode) else "subgrammar"
            rule = RuleNode(name="token_limit", value=subgrammar(inner_name, value))
            rule.max_tokens = max_tokens
    return rule


def with_temperature(value: GrammarNode, temperature: float) -> RuleNode:
    """This sets the sampling temperature to be used for the given portion of the grammar.

    Note that if the grammar passed to us already has some portions with a temperature
    setting in place, those settings will not be overridden.
    """
    if isinstance(value, RuleNode):
        rule = copy.copy(value)
    try:
        rule = RuleNode(name="with_temperature", value=value)
        rule.temperature = temperature
    except ValueError:
        inner_name = value.name if isinstance(value, RuleNode) else "subgrammar"
        rule = RuleNode(name="with_temperature", value=subgrammar(inner_name, value))
        rule.temperature = temperature
    return rule


def capture(value: GrammarNode, name: str) -> RuleNode:
    if isinstance(value, RuleNode):
        return dataclasses.replace(value, capture=name)
    else:
        return RuleNode(name="capture", value=value, capture=name)


def subgrammar(name: str, start: GrammarNode) -> SubgrammarNode:
    return SubgrammarNode(name=name, start=start)


def quote_regex(value: str) -> str:
    return re.sub(r"([\\+*?^$(){}\[\]\.|])", r"\\\1", value)
