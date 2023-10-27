import guidance
from ._byte_range import byte_range 
from ._string import string
from ._zero_or_more import zero_or_more
from .._grammar import Byte
from ._select import select
from types import SimpleNamespace
from pyformlang.regular_expression import PythonRegex


@guidance(stateless=True, dedent=False)
def regex_to_grammar(lm, pattern):
    regex = PythonRegex(pattern)
    cfg = tree_to_grammar(simplify_tree(regex))
    return cfg


def negative_byte_range(forbidden):
    """Given a list of one-char bytes, returns a list of byte ranges that contain every single-character byte except the forbidden ones.
    """
    forb = sorted(set([ord(x) for x in forbidden]))
    start = 0
    ranges = []
    for i in forb:
        if i == 0:
            continue
        newrange = (start, i - 1)
        if newrange[0] < newrange[1]:
            ranges.append(newrange)
        start = i + 1
    if start < 127:
        ranges.append((start, 127))
    ranges = [(i.to_bytes(1, 'big'), j.to_bytes(1, 'big')) for i, j in ranges]
    return ranges

# This is just a helper class so I can merge nodes without having to worry about pyformlang types
class FakeNode:
    def __init__(self, value, sons):
        self.head = SimpleNamespace()
        self.head.value = value
        self.sons = sons

# 
def tree_to_grammar(node):
    """Takes a pyformlang regex tree and returns a guidance gramar
    """
    if not node.sons:
        val = node.head.value
        if val == 'Epsilon' or val == 'Empty':
            val = ''
        else:
            val = string(val)
        return val
    if node.head.value == 'Union':
        vals = [tree_to_grammar(x) for x in node.sons]
        # If select starts with ^ and is a select between bytes, negate it. The way I implemented it is a major hack, should fix
        if all([isinstance(x, Byte) for x in  vals]) and vals[0].byte == b'^':
            vals = [x.byte for x in vals[1:]]
            return select([byte_range(x[0], x[1]) for x in negative_byte_range(vals)])
        return select(vals)
    if node.head.value == 'Concatenation':
        ret = ''
        for x in node.sons:
            ret += tree_to_grammar(x)
        return ret
    if node.head.value == 'Kleene Star':
        return zero_or_more(tree_to_grammar(node.sons[0]))


def simplify_tree(regex):
    """Merges sequence of byte-based concats or unions into single nodes, to make the grammar more compact
    """
    if len(regex.sons) and regex.head.value in ['Concatenation', 'Union']:
        regex = merge_nodes(regex, regex.head.value)
    regex.sons = [simplify_tree(x) for x in regex.sons]
    return regex

def merge_nodes(regex, op):
    """Merges node sequences of concats and unions.
    op must be 'Concatenation' or 'Union'
    """
    assert op in ['Concatenation', 'Union']
    current = regex
    if len(current.sons) and current.head.value == op:
        val = []
        while len(current.sons[0].sons) == 0 and len(current.sons[1].sons) and current.sons[1].head.value == op:
            val.append(current.sons[0].head.value)
            current = current.sons[1]
        if len(current.sons[0].sons) == 0 and len(current.sons[1].sons) == 0 and current.head.value == op:
            val.append(current.sons[0].head.value)
            val.append(current.sons[1].head.value)
            if op == 'Concatenation':
                val = ''.join(val)
                new_node = FakeNode(val, [])
            elif op == 'Union':
                new_node = FakeNode('Union', [FakeNode(x, []) for x in val])
            return new_node
    if len(val) > 1:
        if op == 'Concatenation':
            # merge all chars into a string, return a concat between the string (as a leaf) and whatever right-side children exist
            val = ''.join(val)
            new_node = FakeNode(op, [FakeNode(val, []), merge_nodes(current, op)])
        elif op == 'Union':
            # Merge all select options into a left leaf of a select, whatever operation is left becomes the right child.
            new_node = FakeNode(op, [FakeNode('Union', [FakeNode(x, []) for x in val]), merge_nodes(current, op)])
        return new_node
    else:
        return current
