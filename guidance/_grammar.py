import copy
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
    parse_tags,
)


def string(s: str) -> LiteralNode:
    return LiteralNode(s)


def regex(pattern: str) -> RegexNode:
    return RegexNode(pattern)


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
        rule = RuleNode(name="token_limit", value=value)
    try:
        rule.max_tokens = max_tokens
    except ValueError:
        raise NotImplementedError(
            "token_limit is not implemented for non-terminal RuleNodes"
        )  # TODO: subgrammars
    return rule


def with_temperature(value: GrammarNode, temperature: float) -> RuleNode:
    """This sets the sampling temperature to be used for the given portion of the grammar.

    Note that if the grammar passed to us already has some portions with a temperature
    setting in place, those settings will not be overridden.
    """
    if isinstance(value, RuleNode):
        rule = copy.copy(value)
    else:
        rule = RuleNode(name="with_temperature", value=value)
    try:
        rule.temperature = temperature
    except ValueError:
        raise NotImplementedError(
            "with_temperature is not implemented for non-terminal RuleNodes"
        )  # TODO: subgrammars
    return rule


def capture(value: GrammarNode, name: str) -> RuleNode:
    if isinstance(value, RuleNode):
        return dataclasses.replace(value, capture=name)
    else:
        return RuleNode(name="capture", value=value, capture=name)


def quote_regex(value: str) -> str:
    return re.sub(r"([\\+*?^$(){}\[\]\.|])", r"\\\1", value)
