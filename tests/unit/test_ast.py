from guidance._ast import (
    LarkSerializer,
    RegexNode,
    RuleNode,
)


class TestLarkSerializer:
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
