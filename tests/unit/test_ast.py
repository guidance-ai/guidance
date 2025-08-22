import pydantic
from llguidance import LLMatcher

from guidance._ast import (
    JoinNode,
    JsonNode,
    LarkSerializer,
    LiteralNode,
    RegexNode,
    RepeatNode,
    RuleNode,
    RuleRefNode,
    SelectNode,
    SubgrammarNode,
)
from guidance.library._pydantic import pydantic_to_json_schema


class TestLarkSerializer:
    def test_smoke(self):
        target = LarkSerializer()
        ln = LiteralNode("A")

        result = target.serialize(ln)
        print(result)
        expected = """%llguidance {}

start: START
START: "A"
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_select_literals(self):
        target = LarkSerializer()
        la = LiteralNode("A")
        lb = LiteralNode("B")

        sn = SelectNode((la, lb))

        result = target.serialize(sn)
        print(result)

        expected = """%llguidance {}

start: START
START: "A"
     | "B"

"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_join_literals(self):
        target = LarkSerializer()
        la = LiteralNode("A")
        lb = LiteralNode("B")

        jn = JoinNode((la, lb))

        result = target.serialize(jn)
        print(result)
        expected = """%llguidance {}

start: START
START: "A" "B"
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_named_rule_node(self):
        target = LarkSerializer()
        ren = RegexNode(".*")
        rule_node = RuleNode("my_rule", value=ren)

        result = target.serialize(rule_node)
        print(result)

        expected = """%llguidance {}

start: START
START: MY_RULE
MY_RULE: /.*/
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_capture_rule_node(self):
        target = LarkSerializer()
        ren = RegexNode(".*")
        rule_node = RuleNode("my_rule", value=ren, capture="my_capture")

        result = target.serialize(rule_node)
        print(result)

        expected = """%llguidance {}

start: my_rule
my_rule[capture="my_capture"]: MY_RULE
MY_RULE: /.*/
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_temperature_rule_node(self):
        target = LarkSerializer()
        ren = RegexNode(".*")
        rule_node = RuleNode("my_rule", value=ren, temperature=0.7)

        result = target.serialize(rule_node)
        print(result)

        expected = """%llguidance {}

start: my_rule
my_rule[temperature=0.7]: MY_RULE
MY_RULE: /.*/
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_capture_temperature_rule_node(self):
        target = LarkSerializer()
        ren = RegexNode(".*")
        rule_node = RuleNode("my_rule", value=ren, temperature=0.7, capture="my_capture")

        result = target.serialize(rule_node)
        print(result)

        expected = """%llguidance {}

start: my_rule
my_rule[capture="my_capture", temperature=0.7]: MY_RULE
MY_RULE: /.*/
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_capture_temperature_stop_rule_node(self):
        target = LarkSerializer()
        ren = RegexNode(".*")
        ln = LiteralNode("I'm done")
        rule_node = RuleNode("my_rule", value=ren, temperature=0.7, capture="my_capture", stop=ln)

        result = target.serialize(rule_node)
        print(result)

        expected = """%llguidance {}

start: my_rule
my_rule[capture="my_capture", temperature=0.7, stop="I'm done"]: MY_RULE
MY_RULE: /.*/
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_suffix_rule_node(self):
        target = LarkSerializer()
        ren = RegexNode(".*")
        ln = LiteralNode("I've a suffix")
        rule_node = RuleNode("my_rule", value=ren, suffix=ln)

        result = target.serialize(rule_node)
        print(result)

        expected = """%llguidance {}

start: my_rule
my_rule[suffix="I've a suffix"]: MY_RULE
MY_RULE: /.*/
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_stop_capture_rule_node(self):
        target = LarkSerializer()
        ren = RegexNode(".*")
        ln = LiteralNode("Stopping!")
        rule_node = RuleNode("my_rule", value=ren, stop_capture=r"Stop {} capture!", stop=ln)

        result = target.serialize(rule_node)
        print(result)

        expected = """%llguidance {}

start: my_rule
my_rule[stop="Stopping!", stop_capture="Stop {} capture!"]: MY_RULE
MY_RULE: /.*/
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        # The requirement for stop (or suffix) was only caught on this check
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_nested_rule_node(self):
        target = LarkSerializer()
        ren = RegexNode(r"\d\d")
        rule_node_inner = RuleNode("inner_rule", value=ren, capture="inner_capture")
        ln = LiteralNode("A Literal")
        jn = JoinNode((ln, rule_node_inner))
        rule_node_outer = RuleNode("outer_rule", value=jn)

        result = target.serialize(rule_node_outer)
        print(result)

        expected = """%llguidance {}

start: outer_rule
outer_rule: "A Literal" inner_rule
inner_rule[capture="inner_capture"]: INNER_RULE
INNER_RULE: /\d\d/
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_repeat_node(self):
        target = LarkSerializer()
        ln = LiteralNode("Aa")
        rpt_node = RepeatNode(ln, 1, 23)
        result = target.serialize(rpt_node)
        print(result)

        expected = """%llguidance {}

start: START
START: "Aa"{1,23}
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_repeat_node_zero_or_one(self):
        target = LarkSerializer()
        ln = LiteralNode("Aa")
        rpt_node = RepeatNode(ln, 0, 1)
        result = target.serialize(rpt_node)
        print(result)

        expected = """%llguidance {}

start: START
START: "Aa"?
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_repeat_node_zero_or_more(self):
        target = LarkSerializer()
        ln = LiteralNode("Aa")
        rpt_node = RepeatNode(ln, 0, None)
        result = target.serialize(rpt_node)
        print(result)

        expected = """%llguidance {}

start: START
START: "Aa"*
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_repeat_node_one_or_more(self):
        target = LarkSerializer()
        ln = LiteralNode("Aa")
        rpt_node = RepeatNode(ln, 1, None)
        result = target.serialize(rpt_node)
        print(result)

        expected = """%llguidance {}

start: START
START: "Aa"+
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_repeat_node_two_or_more(self):
        target = LarkSerializer()
        ln = LiteralNode("Aa")
        rpt_node = RepeatNode(ln, 2, None)
        result = target.serialize(rpt_node)
        print(result)

        expected = """%llguidance {}

start: START
START: "Aa"{2,}
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_repeat_node_join_node(self):
        target = LarkSerializer()
        ln1 = LiteralNode("Aa")
        ln2 = LiteralNode("Bb")
        rpt_node = RepeatNode(JoinNode((ln1, ln2)), 1, 4)

        result = target.serialize(rpt_node)

        print(result)
        expected = """%llguidance {}

start: START
START: ("Aa" "Bb"){1,4}
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_repeat_node_select_node(self):
        target = LarkSerializer()
        ln1 = LiteralNode("Aa")
        ln2 = LiteralNode("Bb")
        sn = SelectNode((ln1, ln2))
        rpt_node = RepeatNode(JoinNode((ln1, sn)), 1, 4)

        result = target.serialize(rpt_node)

        print(result)
        expected = """%llguidance {}

start: START
START: ("Aa" ("Aa" | "Bb")){1,4}
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_rule_ref_node(self):
        target = LarkSerializer()
        ln = LiteralNode("Ab")
        rule_node = RuleNode("my_rule", value=ln)
        rref_node = RuleRefNode()
        rref_node.set_target(rule_node)
        base_node = JoinNode((rule_node, rref_node))

        result = target.serialize(base_node)
        print(result)

        expected = """%llguidance {}

start: MY_RULE MY_RULE
MY_RULE: "Ab"
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_json_node(self):
        target = LarkSerializer()

        class Simple(pydantic.BaseModel):
            my_string: str

        schema = pydantic_to_json_schema(Simple)
        jn = JsonNode(schema=schema)
        rule_node = RuleNode("my_rule", value=jn, temperature=0.7)

        result = target.serialize(rule_node)
        print(result)

        expected = """%llguidance {}

start: my_rule

my_rule[temperature=0.7]: %json {
  "properties": {
    "my_string": {
      "title": "My String",
      "type": "string"
    }
  },
  "required": [
    "my_string"
  ],
  "title": "Simple",
  "type": "object"
}

"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err

    def test_subgrammar_node(self):
        target = LarkSerializer()
        ln = LiteralNode("Ab")
        sg_node = SubgrammarNode(ln, skip_regex=r"\d\d")
        rule_node = RuleNode("my_rule", sg_node)
        ren = RegexNode(r"\w+")
        base_node = JoinNode((rule_node, ren))

        result = target.serialize(base_node)
        print(result)

        expected = """%llguidance {}

start: my_rule /\w+/

my_rule: %lark {
%llguidance {}

  start: START
  START: "Ab"

  %ignore /\d\d/
}

"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err
