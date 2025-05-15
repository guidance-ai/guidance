from llguidance import LLMatcher

from guidance._ast import (
    JoinNode,
    LarkSerializer,
    LiteralNode,
    RegexNode,
    RepeatNode,
    RuleNode,
    RuleRefNode,
    SelectNode,
)


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
my_rule[capture="my_capture"]: /.*/
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
my_rule[temperature=0.7]: /.*/
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
my_rule[capture="my_capture", temperature=0.7]: /.*/
"""
        assert result == expected
        grm = LLMatcher.grammar_from_lark(result)
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
inner_rule[capture="inner_capture"]: /\d\d/
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