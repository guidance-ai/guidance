import re
import numpy as np
import pytest
from jsonschema import validate
import json

import guidance
from guidance import (
    gen,
    select,
    optional,
    one_or_more,
    lexeme,
    greedy_grammar,
    gen_grammar,
    lazy_grammar,
)

from ..utils import get_model


@pytest.fixture(scope="module")
def azure_guidance_model(selected_model, selected_model_name):
    if selected_model_name in ["azure_guidance"]:
        return selected_model
    else:
        pytest.skip("Requires Azure Guidance model")


@guidance(stateless=True)
def json_string(lm):
    return lm + lexeme(r'"(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)*"')


@guidance(stateless=True)
def json_number(lm):
    return lm + lexeme(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


@guidance(stateless=True)
def json_value(lm):
    return lm + select(
        [
            json_string(),
            json_number(),
            json_object(),
            json_array(),
            "true",
            "false",
            "null",
        ]
    )


@guidance(stateless=True)
def json_member(lm):
    return lm + json_string() + ":" + json_value()


@guidance(stateless=True)
def json_object(lm):
    return lm + "{" + optional(json_member() + one_or_more("," + json_member())) + "}"


@guidance(stateless=True)
def json_array(lm):
    return lm + "[" + optional(json_value() + one_or_more("," + json_value())) + "]"


@guidance(stateless=True)
def gen_json_object(lm, name: str, max_tokens=100000000):
    grm = greedy_grammar(json_object(), skip_regex=r"[\x20\x0A\x0D\x09]+")
    return lm + gen_grammar(name, grm, no_initial_skip=True, max_tokens=max_tokens)


def test_greedy_json_object(azure_guidance_model):
    lm = azure_guidance_model
    lm += "Three things about J. Random Hacker:\n"
    lm += gen_json_object("hacker", max_tokens=150)
    lm += "\nScore: " + gen("score", regex="[1-3]")
    # make sure it parses as JSON
    obj = json.loads(lm["hacker"])
    assert isinstance(obj, dict)
    assert lm["score"] in ["1", "2", "3"]
