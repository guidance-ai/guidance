import pyparsing as pp

pp.ParserElement.enable_packrat()
# pp.enable_diag(pp.Diagnostics.enable_debug_on_named_expressions)
# pp.autoname_elements()

class SavedTextNode:
    """A node that saves the text it matches."""
    def __init__(self, s, loc, tokens):
        start_pos = tokens[0]
        if len(tokens) == 3:
            end_pos = tokens[2]
        else:
            end_pos = loc
        self.text = s[start_pos:end_pos]
        assert len(tokens[1]) == 1
        self.tokens = tokens[1][0]
    def __repr__(self):
        return "SavedTextNode({})".format(self.text) + self.tokens.__repr__()
    def __getitem__(self, item):
        return self.tokens[item]
    def __len__(self):
        return len(self.tokens)
    def get_name(self):
        return self.tokens.get_name()
    def __contains__(self, item):
        return item in self.tokens
    def __getattr__(self, name):
        return getattr(self.tokens, name)
    def __call__(self, *args, **kwds):
        return self.tokens(*args, **kwds)
def SavedText(node):
    return pp.Located(node).add_parse_action(SavedTextNode)

program = pp.Forward()
program_chunk = pp.Forward()

## whitespace ##

ws = pp.White()
opt_ws = pp.Optional(ws)


## comments ##

# long-form comments {{!-- my comment --}}
command_end = pp.Suppress(opt_ws + "}}") | pp.Suppress(opt_ws + "~}}" + opt_ws)
long_comment_start = pp.Suppress(pp.Literal("{{") + pp.Optional("~") + pp.Literal("!--"))
long_comment_end =  pp.Suppress(pp.Literal("--") + command_end)
not_long_comment_end = "-" + ~pp.FollowedBy("-}}") + ~pp.FollowedBy("-~}}")
long_comment_content = not_long_comment_end | pp.OneOrMore(pp.CharsNotIn("-"))
long_comment = SavedText(pp.Group(pp.Combine(long_comment_start + pp.ZeroOrMore(long_comment_content) + long_comment_end))("long_comment").set_name("long_comment"))

# short-form comments  {{! my comment }}
comment_start = pp.Suppress("{{" + pp.Optional("~") + "!")
not_comment_end = "}" + ~pp.FollowedBy("}") | "~" + ~pp.FollowedBy("}}")
comment_content = not_comment_end | pp.OneOrMore(pp.CharsNotIn("~}"))
comment = SavedText(pp.Group(pp.Combine(comment_start + pp.ZeroOrMore(comment_content) + command_end))("comment"))


## literals ##

literal = pp.Forward().set_name("literal")

# basic literals
string_literal = pp.Group(pp.Suppress('"') + pp.ZeroOrMore(pp.CharsNotIn('"')) + pp.Suppress('"') | pp.Suppress("'") + pp.ZeroOrMore(pp.CharsNotIn("'")) + pp.Suppress("'"))("string_literal")
number_literal = pp.Group(pp.Word(pp.srange("[0-9.]")))("number_literal")
boolean_literal = pp.Group("True" | pp.Literal("False"))("boolean_literal")

# object literal
object_literal = pp.Forward().set_name("object_literal")
object_start = pp.Suppress("{")
object_end = pp.Suppress("}")
empty_object = object_start + object_end
object_item = string_literal + pp.Suppress(":") + literal
single_item_object = object_start + object_item + object_end
object_sep = pp.Suppress(",")
multi_item_object = object_start + object_item + pp.ZeroOrMore(object_sep + object_item) + object_end
object_literal <<= pp.Group(empty_object | single_item_object | multi_item_object)("object_literal")

# array literal
array_literal = pp.Forward().set_name("array_literal")
array_start = pp.Suppress("[")
array_end = pp.Suppress("]")
array_item = literal
empty_array = array_start + array_end
single_item_array = array_start + array_item + array_end
array_sep = pp.Suppress(",")
multi_item_array = array_start + array_item + pp.ZeroOrMore(array_sep + array_item) + array_end
array_literal <<= pp.Group(empty_array | single_item_array | multi_item_array)("array_literal")

literal <<= string_literal | number_literal | boolean_literal | array_literal | object_literal


## infix operators ##

code_chunk_no_infix = pp.Forward().set_name("code_chunk_no_infix")

class OpNode:
    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.operator)
    def __getitem__(self, item):
        return getattr(self, item)
    def get_name(self):
        return self.name

class UnOp(OpNode):
    def __init__(self, tokens):
        self.operator = tokens[0][0]
        self.value = tokens[0][1]
        self.name = "unary_operator"

class BinOp(OpNode):
    def __init__(self, tokens):
        self.operator = tokens[0][1]
        self.lhs = tokens[0][0]
        self.rhs = tokens[0][2]
        self.name = "binary_operator"

infix_operator_block = pp.infix_notation(code_chunk_no_infix, [
    (pp.one_of('- not'), 1, pp.OpAssoc.RIGHT, UnOp),
    (pp.one_of('* /'), 2, pp.OpAssoc.LEFT, BinOp),
    (pp.one_of('+ -'), 2, pp.OpAssoc.LEFT, BinOp),
    (pp.one_of('< > <= >= == != is in'), 2, pp.OpAssoc.LEFT, BinOp),
    (pp.one_of('and'), 2, pp.OpAssoc.LEFT, BinOp),
    (pp.one_of('or'), 2, pp.OpAssoc.LEFT, BinOp),
])


## commands ##

code_chunk = pp.Forward().set_name("code_chunk")
not_keyword = ~pp.FollowedBy(pp.Keyword("or") | pp.Keyword("else") | pp.Keyword("elif") | pp.Keyword("not"))
command_name = pp.Combine(not_keyword + pp.Word(pp.srange("[@A-Za-z_]"), pp.srange("[A-Za-z_0-9\.]")))
variable_name = pp.Word(pp.srange("[@A-Za-z_]"), pp.srange("[A-Za-z_0-9]"))
variable_ref = not_keyword + pp.Group(pp.Word(pp.srange("[@A-Za-z_]"), pp.srange("[@A-Za-z_0-9\.\[\]\"'-]")))("variable_ref").set_name("variable_ref")
keyword = pp.Group(pp.Keyword("break") | pp.Keyword("continue"))("keyword")

# command arguments
command_arg = pp.Forward()
named_command_arg = variable_name + "=" + code_chunk
command_arg <<= pp.Group(named_command_arg)("named_command_arg").set_name("named_command_arg") | pp.Group(code_chunk)("positional_command_arg").set_name("positional_command_arg")

# whitespace command format {{my_command arg1 arg2=val}}
ws_command_call = pp.Forward().set_name("ws_command_call")
command_arg_and_ws = pp.Suppress(ws) + command_arg
ws_command_args = pp.OneOrMore(command_arg_and_ws)
# note that we have to list out all the operators here because we match before the infix operator checks
ws_command_call <<= command_name("name") + ~pp.FollowedBy(pp.one_of("+ - * / or not is and <= == >= != < >")) + ws_command_args

# paren command format {{my_command(arg1, arg2=val)}}
paren_command_call = pp.Forward().set_name("paren_command_call")
command_arg_and_comma_ws = pp.Suppress(",") + command_arg
paren_command_args = pp.Optional(command_arg) + pp.ZeroOrMore(command_arg_and_comma_ws)
paren_command_call <<= (command_name("name") + pp.Suppress("(")).leave_whitespace() - paren_command_args + pp.Suppress(")")

# code chunks
enclosed_code_chunk = pp.Forward().set_name("enclosed_code_chunk")
paren_group = (pp.Suppress("(") - enclosed_code_chunk + pp.Suppress(")")).set_name("paren_group")
enclosed_code_chunk_cant_infix = (pp.Group(ws_command_call)("command_call") | pp.Group(paren_command_call)("command_call") | literal | keyword | variable_ref | paren_group) + ~pp.FollowedBy(pp.one_of("+ - * / or not is and <= == >= != < >"))
enclosed_code_chunk <<= enclosed_code_chunk_cant_infix | infix_operator_block
code_chunk_no_infix <<= (paren_group | pp.Group(paren_command_call)("command_call") | literal | keyword | variable_ref) # used by infix_operator_block
code_chunk_cant_infix = code_chunk_no_infix + ~pp.FollowedBy(pp.one_of("+ - * / or not is and <= == >= != < >")) # don't match infix operators so we can run this before infix_operator_block
code_chunk_cant_infix.set_name("code_chunk_cant_infix")
code_chunk <<= code_chunk_cant_infix | infix_operator_block

# command/variable
command_start = pp.Suppress("{{" + ~pp.FollowedBy("!") + pp.Optional("~"))
simple_command_start = pp.Suppress("{{" + ~pp.FollowedBy("!") + pp.Optional("~")) + ~pp.FollowedBy(pp.one_of("# / >"))
command = SavedText(pp.Group(simple_command_start + enclosed_code_chunk + command_end)("command"))

# partial
always_call = pp.Group(paren_command_call | command_name("name") + pp.Optional(ws_command_args))
partial = pp.Group(pp.Suppress(pp.Combine(command_start + ">")) + always_call("command_call") + command_end)("partial")

# block command {{#my_command arg1 arg2=val}}...{{/my_command}}
separator = pp.Group(pp.Keyword("or") | pp.Keyword("else") | (pp.Keyword("elif") + ws_command_args))("separator").set_name("separator")
block_command = pp.Forward()
block_command_call = always_call("command_call")
block_command_open = pp.Suppress(pp.Combine(command_start + "#")) + block_command_call + command_end
block_command_sep = (command_start + separator + command_end)("block_command_sep").set_name("block_command_sep")
block_command_close = SavedText(pp.Group(command_start + pp.Suppress("/") + command_name + command_end)("block_command_close").set_name("block_command_close"))
block_command_content = (pp.Group(program)("block_content_chunk") + pp.ZeroOrMore(block_command_sep + pp.Group(program)("block_content_chunk"))).set_name("block_content")
block_command <<= (block_command_open + SavedText(pp.Group(block_command_content)("block_content")) + block_command_close).leave_whitespace()
block_command = SavedText(pp.Group(block_command)("block_command")).set_name("block_command")

# block partial {{#>my_command arg1 arg2=val}}...{{/my_command}}
block_partial = pp.Forward()
block_partial_call = always_call("command_call")
block_partial_open = pp.Combine(command_start + pp.Suppress("#>")) + block_partial_call + command_end
block_partial_close = command_start + pp.Suppress("/") + command_name + command_end
block_partial <<= block_partial_open + program + pp.Suppress(block_partial_close)
block_partial = SavedText(pp.Group(block_partial)("block_partial"))

# escaped commands \{{ not a command }}
not_command_end = "}" + ~pp.FollowedBy("}")
escaped_command = SavedText(pp.Group(pp.Suppress("\\") + command_start + pp.Combine(pp.ZeroOrMore(pp.CharsNotIn("}") | not_command_end)) + command_end)("escaped_command"))
unrelated_escape = "\\" + ~pp.FollowedBy(command_start)


## content ##

not_command_start = "{" + ~pp.FollowedBy("{" + pp.CharsNotIn("{"))
not_command_escape = "\\" + ~pp.FollowedBy("{{")
stripped_whitespace = pp.Suppress(pp.Word(" \t\r\n")) + pp.FollowedBy("{{~")
unstripped_whitespace = pp.Word(" \t\r\n") # no need for a negative FollowedBy because stripped_whitespace will match first
content = pp.Group(pp.Combine(pp.OneOrMore(stripped_whitespace | unstripped_whitespace | not_command_start | not_command_escape | pp.CharsNotIn("{\\ \t\r\n"))))("content").set_name("content")

# keyword_command = SavedText(pp.Group(command_start + keyword + ws_command_args + command_end)("keyword_command"))
# block_content_chunk = long_comment | comment | escaped_command | unrelated_escape | block_partial | block_command | partial | command | content
# block_content <<= pp.ZeroOrMore(block_content_chunk)("program").leave_whitespace()

## global program ##

program_chunk <<= (long_comment | comment | escaped_command | unrelated_escape | block_partial | block_command | partial | command | content).leave_whitespace()
program <<= pp.ZeroOrMore(program_chunk)("program").leave_whitespace().set_name("program")
grammar = (program + pp.StringEnd()).parse_with_tabs()