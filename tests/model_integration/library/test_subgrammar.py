import json
import re

import guidance
from guidance import (
    gen,
    one_or_more,
    optional,
    select,
)
from guidance.library._subgrammar import lexeme, subgrammar


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
    grm = subgrammar(body=json_object(), name=name, skip_regex=r"[\x20\x0A\x0D\x09]+", max_tokens=max_tokens)
    return lm + grm


def test_greedy_json_object(selected_model: guidance.models.Model):
    lm = selected_model
    lm += "John Doe's name, age, and birthday:\n"
    lm += gen_json_object("hacker", max_tokens=1000)
    lm += "\nScore: " + gen("score", regex="[1-3]")
    # make sure it parses as JSON
    obj = json.loads(lm["hacker"])
    assert isinstance(obj, dict)
    assert lm["score"] in ["1", "2", "3"]


def test_greedy_single_terminal(selected_model: guidance.models.Model):
    lm = selected_model
    lm += "A number: "
    lm += subgrammar(body=lexeme(r"[0-9]{3}"))
    assert re.search(r": [0-9]{3}$", str(lm))
