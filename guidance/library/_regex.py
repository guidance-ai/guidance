import guidance
from .._grammar import byte_range, select, string, Byte
from ._zero_or_more import zero_or_more
from ._any_char_but import any_char_but
import re
from types import SimpleNamespace
from pyformlang.regular_expression import PythonRegex


@guidance(stateless=True, dedent=False)
def regex(lm, pattern):
    # find all of the brackets we'll need to negate later
    nots = re.findall('\[\^(.*?)\]', pattern)
    nots = [re.compile('[' + x + ']') for x in nots]
    # Compensating for a weird design choice in pyformlang where they don't accept \n in .
    pattern = re.sub(r'(?<!\\)\.', '(.|\n)', pattern)
    regex = PythonRegex(pattern)
    cfg = tree_to_grammar(simplify_tree(regex), nots=nots)
    return cfg


# This is just a helper class so I can merge nodes without having to worry about pyformlang types
class FakeNode:
    def __init__(self, value, sons):
        self.head = SimpleNamespace()
        self.head.value = value
        self.sons = sons
    def get_tree_str(self, depth: int = 0) -> str:
        """ Get a string representation of the tree behind the regex
        """
        temp = " " * depth + str(self.head) + "\n"
        for son in self.sons:
            temp += son.get_tree_str(depth + 1)
        return temp


# 
def tree_to_grammar(node, nots):
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
        vals = [tree_to_grammar(x, nots) for x in node.sons]
        # If select starts with ^ and is a select between bytes and it is listed in our negations, negate it. The way I implemented it is a major hack, should fix
        if all([isinstance(x, Byte) for x in vals]) and vals[0].byte == b'^':
            temp_vals = [x.byte for x in vals[1:]]
            all_chars = b''.join(temp_vals).decode('utf8')
            for notz in nots:
                # The not matches all of the chars.
                # TODO: Technically there could be a situation where this is true AND this is not actually what we had in a regex, so this code is wrong. But it's probably never going to happen, and we'll replace this code later.
                if all([notz.match(x) is not None for x in all_chars]):
                    return any_char_but(temp_vals)

            # if b''.join(temp_vals).decode('utf8') in nots:
            #     # print('Negating', temp_vals)
            #     return select([byte_range(x[0], x[1]) for x in negative_byte_range(temp_vals)])
        return select(vals)
    if node.head.value == 'Concatenation':
        ret = ''
        for x in node.sons:
            ret += tree_to_grammar(x, nots)
        return ret
    if node.head.value == 'Kleene Star':
        return zero_or_more(tree_to_grammar(node.sons[0], nots))


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
        to_visit = [current]
        while to_visit:
            current = to_visit.pop()
            leaves = [x for x in current.sons if len(x.sons) == 0]
            ops = [x for x in current.sons if len(x.sons) and x.head.value == op]
            if len(leaves) + len(ops) == len(current.sons):
                val.extend([x.head.value for x in leaves])
                to_visit += [x for x in current.sons if len(x.sons) and x.head.value == op]
        # if all I have is leaves, I can group them
        if all([len(x.sons) == 0 for x in current.sons]) and current.head.value == op:
            # val.extend([x.head.value for x in current.sons])
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
        return regex
