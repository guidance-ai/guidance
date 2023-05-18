import parsimonious

# define the Guidance language grammar
grammar = parsimonious.grammar.Grammar(
r"""
template = template_chunk*
template_chunk = comment / slim_comment / escaped_command / unrelated_escape / command / command_block / content

comment = comment_start comment_content* comment_end
comment_start = "{{!--"
comment_content = not_comment_end / ~r"[^-]*"
not_comment_end = "-" !"-}}"
comment_end = "--}}"

slim_comment = slim_comment_start slim_comment_content* slim_comment_end
slim_comment_start = "{{" "~"? "!"
slim_comment_content = not_slim_comment_end / ~r"[^}]*"
not_slim_comment_end = "}" !"}"
slim_comment_end = "}}"

command = command_start command_content command_end
command_block = command_block_open template (command_block_sep template)* command_block_close
command_block_open = command_start "#" block_command_call command_end
command_block_sep = command_start ("or" / "else") command_end
command_block_close = command_start "/" command_name command_end
command_start = "{{" !"!" "~"?
not_command_start = "{" !"{"
not_command_escape = "\\" !"{{"
command_end = "~"? "}}"
command_contents = ~'[^{]*'
block_command_call = command_name command_args?
command_content = command_call / variable_ref
command_call = command_name command_args
command_args = command_arg_and_ws+
command_arg_and_ws = ws command_arg
command_arg = named_command_arg / positional_command_arg
positional_command_arg = command_arg_group / literal / variable_ref
named_command_arg = variable_name "=" (literal / variable_ref)
command_arg_group = "(" command_content ")"
ws = ~r'\s+'
command_contentasdf = ~"[a-z 0-9]*"i
command_name = ~r"[a-z][a-z_0-9\.]*"i / "<" / ">" / "==" / "!=" / ">=" / "<="
variable_ref = !"or" !"else" ~r"[@a-z][a-z_0-9\.\[\]\"'-]*"i
variable_name = ~r"[@a-z][a-z_0-9]*"i
contentw = ~r'.*'
content = (not_command_start / not_command_escape / ~r"[^{\\]")+
unrelated_escape = "\\" !command_start
escaped_command = "\\" command_start ~r"[^}]*" command_end

literal = string_literal / number_literal / boolean_literal / array_literal / object_literal

string_literal = ~r'"[^\"]*"' / ~r"'[^\']*'"

number_literal = ~r"[0-9\.]+"

boolean_literal = "True" / "False"

array_literal = empty_array / single_item_array / multi_item_array
empty_array = array_start ws? array_end
single_item_array = array_start ws? array_item ws? array_end
array_sep = ws? "," ws?
multi_item_array = array_start ws? array_item (array_sep array_item)* ws? array_end
array_start = "["
array_end = "]"
array_item = literal

object_literal = empty_object / single_item_object / multi_item_object
empty_object = object_start ws? object_end
single_item_object = object_start ws? object_item ws? object_end
object_sep = ws? "," ws?
multi_item_object = object_start ws? object_item (object_sep object_item)* ws? object_end
object_start = "{"
object_end = "}"
object_item = string_literal ws? ":" ws? literal
""")