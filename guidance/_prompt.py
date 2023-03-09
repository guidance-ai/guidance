import ast
import inspect
import re
import html
import uuid
import sys
import parsimonious
import warnings
import copy
from .llms import _openai
from . import _utils
import guidance

class PromptCompletion:
    ''' Represents the result of an executed prompt.
    '''
    def __init__(self, variables, text, text_html, prompt):
        self.variables = variables
        self.text = text
        self.text_html = text_html
        self.prompt = prompt

    def __getitem__(self, key):
        return self.variables[key]

    def __repr__(self):
        return self.text

    def _repr_html_(self):
        return self.text_html
    
    def __str__(self) -> str:
        return self.text

class Prompt:
    ''' A prompt template that can be compiled and executed to generate a PromptCompletion result.
    '''

    def __init__(self, template, call_function=None, llm=None, echo=False, cache_seed=0, logprobs=None, **kwargs):
        """ Create a new Prompt object from a prompt string.
        """

        # see if we were given a raw function instead of a string template
        if not isinstance(template, str):
            if callable(template):
                sig = inspect.signature(template)
                args = ""
                for name,_ in sig.parameters.items():
                    args += f" {name}={name}"
                fname = _utils.find_func_name(kwargs)
                kwargs[fname] = template
                template = "{{set (%s%s)}}" % (fname, args)

        self._template = template
        self.call_function = call_function
        self.llm = llm
        self.echo = echo
        self.cache_seed = cache_seed
        self.logprobs = logprobs
        self.default_vars = kwargs

        # find all the handlebars-style partial inclusion tags and replace them with the partial template
        def replace_partial(match):
            partial_name,args_string = match.group(1).split(" ", 1)
            if partial_name not in kwargs:
                raise ValueError("Partial '%s' not given in the keyword args:" % partial_name)
            out = "{{#block '"+partial_name+"'"
            if len(args_string) > 0:
                out += " " + args_string
            out += "}}" + kwargs[partial_name]._template + "{{/block}}"
            self.default_vars = {**kwargs[partial_name].default_vars, **self.default_vars} # pull in the default vars from the partial
            return out
        self._template = re.sub(r"{{>(.*?)}}", replace_partial, self._template)

        self.patch_stack = []

        # default to an OpenAI llm
        if self.llm is None:
            self.llm = guidance.llm

        # if we don't have a custom call function, we parse the string
        if call_function is None:
            self.tree = grammar.parse(self._template)

    # def __getattribute__(self, name):
    #     if name == "template":
    #         patched_kwargs = {k: v for k, v in kwargs.items()}
    #         if "template" not in kwargs:
    #             patched_kwargs["template"] = self._template
    #         for patch, arg_names in self.patch_stack:
    #             tmp = patch(**{k: patched_kwargs[k] for k in arg_names})
    #             if len(arg_names) == 1:
    #                 tmp = (tmp,)
    #             for k, v in zip(arg_names, tmp):
    #                 patched_kwargs[k] = v
    
    def patch(self, patch_functions):
        """ Add a patch to this prompt.
        """

        # copy ourself
        new_self = copy.copy(self)

        if callable(patch_functions):
            patch_functions = []
        for patch_function in patch_functions:
            arg_names = inspect.getfullargspec(patch_function).args
            new_self.patch_stack.append((patch_function, arg_names))
        
        return new_self
    
    def __repr__(self):
        return self._template

    def _repr_html_(self):
        display_out = self._template
        # add syntax highlighting
        display_out = re.sub(r"(\{\{generate.*?\}\})", r"<span style='background-color: rgba(0, 165, 0, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
        display_out = re.sub(r"(\{\{#select\{\{/select.*?\}\})", r"<span style='background-color: rgba(0, 165, 0, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
        display_out = re.sub(r"(\{\{#each [^'\"].*?\{\{/each.*?\}\})", r"<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
        display_out = re.sub(r"(\{\{(?!generate)(?!#select)(?!#each)(?!/each)(?!/select).*?\}\})", r"<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
        display_out = "<pre style='padding: 7px; border-radius: 4px; background: white; white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 16px; line-height: 24px; color: #000'>"+display_out+"</pre>"
        return display_out
    
    def __call__(self, **kwargs):

        # merge in the default variables
        kwargs = {**self.default_vars, **kwargs}

        # apply any patches
        patched_kwargs = {k: v for k, v in kwargs.items()}
        if "template" not in kwargs:
            patched_kwargs["template"] = self._template
        for patch, arg_names in self.patch_stack:
            tmp = patch(**{k: patched_kwargs[k] for k in arg_names})
            if len(arg_names) == 1:
                tmp = (tmp,)
            for k, v in zip(arg_names, tmp):
                patched_kwargs[k] = v
        if "template" not in kwargs and (patched_kwargs["template"] == self._template) and self.call_function == None:
            del patched_kwargs["template"]

        # if we have a custom call function, we call that instead
        if self.call_function is not None:
            return self.call_function(**patched_kwargs)
        
        # ...otherwise we do the normal parsing
        built_ins = {
            "generate": _generate,
            "each": _each,
            "select": _select,
            "if": _if,
            "unless": _unless,
            "add": _add,
            "subtract": _subtract,
            "strip": _strip,
            "block": _block,
            "set": _set
        }
        variables = {}
        variables.update(built_ins)
        variables.update(patched_kwargs)

        vi = TopDownVisitor(variables, self)

        output = vi.visit(self.tree)
        
        # remove the built-ins from the variables we return
        for k in built_ins:
            del variables[k]

        def start_generate_or_select(x):
            no_echo = "echo=False" in x.group(1)
            alpha = 1.0 if no_echo else 1.0
            
            # script that toggles the viisibility of the next element
            click_script = 'var e = this.nextElementSibling; if (e.style.display == "inline") { e.style.display = "none"; this.style.borderRight = "1px solid rgb(0, 165, 0, 1)"; } else { e.style.display = "inline"; this.style.borderRight = "0px";}'

            if no_echo:
                out = f'''<div style='background-color: rgba(0, 165, 0, 0.25); border-radius: 4px 0px 0px 4px; border: 1px solid rgba(0, 165, 0, 1); padding-left: 3px; padding-right: 3px; user-select: none; color: rgb(0, 165, 0, 1.0); display: inline; font-weight: normal; cursor: pointer' onClick='{click_script}'>no echo</div>'''
                out += "<span style='background-color: rgba(0, 165, 0, 0.25); opacity: {}; display: none;' title='{}'>".format(alpha, x.group(1))
            else:
                out = "<span style='background-color: rgba(0, 165, 0, 0.25); opacity: {}; display: inline;' title='{}'>".format(alpha, x.group(1))
            return out
        
        def start_each(x):
            no_echo = "echo=False" in x.group(1)
            alpha = 0.5 if no_echo else 1.0
            color = "rgba(0, 138.56128016, 250.76166089, 0.25)" if "each '" not in x.group(1) and "each \"" not in x.group(1) else "rgba(0, 165, 0, 0.25)"
            return "<span style='opacity: {}; display: inline; background-color: {};' title='{}'>".format(alpha, color, x.group(1))

        display_out = html.escape(output)

        # format the generate command results
        display_out = re.sub(r"__GMARKER_START_generate\$([^\$]*)\$___", start_generate_or_select, display_out)
        display_out = display_out.replace("__GMARKER_END_generate$$___", "</span>")
        def click_loop_start(id, total_count, echo, color):
            # echo = x.group(1) == "True"
            # total_count = int(x.group(2))
            # id = x.group(3)
            click_script = '''
function cycle_IDVAL(button_el) {
    var i = 0;
    while (i < 50) {
        var el = document.getElementById("IDVAL_" + i);
        if (el.style.display == "inline") {
            el.style.display = "none";
            var next_el = document.getElementById("IDVAL_" + (i+1));
            if (!next_el) {
                next_el = document.getElementById("IDVAL_0");
            }
            if (next_el) {
                next_el.style.display = "inline";
            }
            break;
        }
        i += 1;
    }
    button_el.innerHTML = (((i+1) % TOTALCOUNT) + 1)  + "/" + TOTALCOUNT;
}
cycle_IDVAL(this);'''.replace("IDVAL", id).replace("TOTALCOUNT", str(total_count)).replace("\n", "")
            out = f'''<div style='background: rgba(255, 255, 255, 0.0); border-radius: 4px 0px 0px 4px; border: 1px solid {color}; border-right: 0px; padding-left: 3px; padding-right: 3px; user-select: none; color: {color}; display: inline; font-weight: normal; cursor: pointer' onClick='{click_script}'>1/{total_count}</div>'''
            out += f"<div style='display: inline;' id='{id}_0'>"
            return out
        def click_loop_mid(id, index, echo):
            # echo = x.group(1) == "True"
            # index = int(x.group(2))
            # id = x.group(3)
            alpha = 1.0 if not echo else 0.5
            out = f"</div><div style='display: none; opacity: {alpha}' id='{id}_{index}'>"
            return out
        display_out = re.sub(
            r"__GMARKER_generate_many_start_([^_]+)_([0-9]+)\$([^\$]*)\$___",
            lambda x: click_loop_start(x.group(3), int(x.group(2)), x.group(1) == "True", "rgb(0, 165, 0, 1)"),
            display_out
        )
        display_out = re.sub(
            r"__GMARKER_generate_many_([^_]+)_([0-9]+)\$([^\$]*)\$___",
            lambda x: click_loop_mid(x.group(3), int(x.group(2)), x.group(1) == "True"),
            display_out
        )
        display_out = re.sub(r"__GMARKER_generate_many_end\$([^\$]*)\$___", "</div>", display_out)

        # format the each command results
        display_out = re.sub(r"__GMARKER_START_each\$([^\$]*)\$___", start_each, display_out)
        display_out = re.sub(
            r"__GMARKER_each_noecho_start_([^_]+)_([0-9]+)\$([^\$]*)\$___",
            lambda x: click_loop_start(x.group(3), int(x.group(2)), False, "rgb(100, 100, 100, 1)"),
            display_out
        )
        display_out = re.sub(
            r"__GMARKER_each_noecho_([^_]+)_([0-9]+)\$([^\$]*)\$___",
            lambda x: click_loop_mid(x.group(3), int(x.group(2)), False),
            display_out
        )
        display_out = re.sub(r"__GMARKER_each_noecho_end\$([^\$]*)\$___", "</div>", display_out)
        
        # format the set command results
        display_out = re.sub(r"__GMARKER_set\$([^\$]*)\$___", r"<div style='background-color: rgba(165, 165, 165, 0); border-radius: 4px 4px 4px 4px; border: 1px solid rgba(165, 165, 165, 1); border-left: 2px solid rgba(165, 165, 165, 1); border-right: 2px solid rgba(165, 165, 165, 1); padding-left: 0px; padding-right: 3px; color: rgb(165, 165, 165, 1.0); display: inline; font-weight: normal; overflow: hidden;'><div style='display: inline; background: rgba(165, 165, 165, 1); padding-right: 5px; padding-left: 4px; margin-right: 3px; color: #fff'>set</div>\1</div>", display_out)
        display_out = re.sub(r"__GMARKER_START_set\$([^\$]*)\$___", r"<span style='display: inline;' title='\1'>", display_out)

        display_out = re.sub(r"__GMARKER_START_select\$([^\$]*)\$___", start_generate_or_select, display_out)
        display_out = display_out.replace("__GMARKER_END_select$$___", "</span>")
        display_out = re.sub(r"__GMARKER_START_variable_ref\$([^\$]*)\$___", r"<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='\1'>", display_out)
        display_out = display_out.replace("__GMARKER_END_variable_ref$$___", "</span>")
        display_out = display_out.replace("__GMARKER_each$$___", "<div style='border-left: 1px dashed rgb(0, 0, 0, .2); border-top: 0px solid rgb(0, 0, 0, .2); margin-right: -4px; display: inline; width: 4px; height: 24px;'></div>")
        display_out = re.sub(r"__GMARKER_START_block\$([^\$]*)\$___", r"<span style='background-color: rgba(165, 165, 165, 0.15); display: inline;' title='\1'>", display_out)
        display_out = re.sub(r"__GMARKER_START_([^\$]*)\$([^\$]*)\$___", r"<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='\2'>", display_out)
        # display_out = re.sub(r"__GMARKER_START_([^\$]*)\$([^\$]*)\$___", r"<span style='background-color: rgba(165, 165, 165, 0.25); display: inline;' title='\2'>", display_out)
        display_out = re.sub(r"__GMARKER_END_([^\$]*)\$\$___", "</span>", display_out)
        display_out = "<pre style='padding: 7px; border-radius: 4px; background: white; white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 16px; line-height: 24px; color: #000'>"+display_out+"</pre>"

        # strip out the markers for the unformatted output
        output = strip_markers(output)

        return PromptCompletion(variables, output, display_out, self)

def strip_markers(s):
    return re.sub(r"__GMARKER_([^\$]*)\$([^\$]*)\$___", r"", s, flags=re.MULTILINE | re.DOTALL)

grammar = parsimonious.grammar.Grammar(
r"""
template = template_chunk*
template_chunk = escaped_command / unrelated_escape / command / command_block / content
command = command_start command_content command_end
command_block = command_block_open template (command_block_sep template)* command_block_close
command_block_open = command_start "#" block_command_call command_end
command_block_sep = command_start ("or" / "else") command_end
command_block_close = command_start "/" command_name command_end
command_start = "{{" "~"?
not_command_start = "{" !"{"
not_command_escape = "\\" !"{{"
command_end = "~"? "}}"
command_contents = ~'[^{]*'
block_command_call = command_name command_args
command_content = command_call / variable_ref
command_call = command_name command_args
command_args = command_arg_and_ws+
command_arg_and_ws = ws command_arg
command_arg = named_command_arg / positional_command_arg
positional_command_arg = command_arg_group / variable_ref / literal
named_command_arg = variable_name "=" (variable_ref / literal)
command_arg_group = "(" command_content ")"
ws = ~r'\s+'
command_contentasdf = ~"[a-z 0-9]*"i
command_name = ~r"[a-z][a-z_0-9\.]*"i
variable_ref = !"or" !"else" ~r"[@a-z][a-z_0-9\.]*"i
variable_name = ~r"[@a-z][a-z_0-9\.]*"i
contentw = ~r'.*'
content = not_command_start / not_command_escape / ~r"[^{\\]*"
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

class PositionalArgument:
    def __init__(self, value):
        self.value = value

class NamedArgument:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class TopDownVisitor():
    def __init__(self, variables, prompt_object, logprobs=None):
        self.variable_stack = [variables]
        self.prompt_object = prompt_object
        self.logprobs = logprobs
        self.prefix = ''
        self.prefix_tokens = []
        self.block_content = []
    
    def visit(self, node, next_node=None, prev_node=None):

        if node.expr_name == 'variable_ref':
            var = self.get_variable(node.text)

            if callable(var):
                sig = inspect.signature(var)
                kwargs = {}
                if "template_context" in sig.parameters:
                    template_context = {}
                    if len(self.block_content[-1]) == 1:
                        template_context["@block_text"] = self.block_content[-1][0].text
                    if hasattr(self.prompt_object, "tokenizer"):
                        template_context["@tokenizer"] = self.prompt_object.tokenizer
                    kwargs["template_context"] = template_context
                var = var(**kwargs)
            
            return var

        elif node.expr_name == 'variable_name':
            return node.text

        elif node.expr_name == 'content':
            self._extend_prefix(node.text)
            return node.text

        elif node.expr_name == 'command_args':
            visited_children = [self.visit(child) for child in node.children]
            return visited_children

        elif node.expr_name == 'command_arg_and_ws':
            visited_children = [self.visit(child) for child in node.children]
            return visited_children[1]

        elif node.expr_name == 'positional_command_arg':
            visited_children = [self.visit(child) for child in node.children]
            return PositionalArgument(visited_children[0])

        elif node.expr_name == 'named_command_arg':
            visited_children = [self.visit(child) for child in node.children]
            return NamedArgument(visited_children[0], visited_children[2])

        elif node.expr_name == 'command_name':
            return node.text

        elif node.expr_name == 'escaped_command':
            self._extend_prefix(node.text[1:])
            return node.text[1:]

        elif node.expr_name == 'literal':
            return ast.literal_eval(node.text)

        elif node.expr_name == 'command':
            self.block_content.append([])
            visited_children = [self.visit(child, next_node, prev_node) for child in node.children]
            self.block_content.pop()
            out = "".join("" if c is None else c for c in visited_children)
            
            command_head = node.children[1].children[0]
            if command_head.expr_name == 'variable_ref':
                self._extend_prefix(out)
                name = "variable_ref"
            elif command_head.expr_name == 'command_call':
                name = command_head.children[0].text
            else:
                raise Exception("Unknown command head type: "+command_head.expr_name)

            node_text = node.text.replace("$", "DOLLAR_SIGN")
            return f"__GMARKER_START_{name}${node_text}$___{out}__GMARKER_END_{name}$$___"

        elif node.expr_name == 'command_arg_group':
            visited_children = [self.visit(child) for child in node.children]
            return visited_children[1]

        elif node.expr_name == 'command_call':
            visited_children = [self.visit(child) for child in node.children]
            command_name, args = visited_children


            # merge list of dicts into one dict
            # merged_variables = {k: v for d in reversed(self.variable_stack) for k, v in d.items()}

            if self.variable_exists(command_name):
                command_function = self.get_variable(command_name)
                positional_args = []
                named_args = {}
                for arg in args:
                    if isinstance(arg, PositionalArgument):
                        positional_args.append(arg.value)
                    elif isinstance(arg, NamedArgument):
                        named_args[arg.name] = arg.value
                sig = inspect.signature(command_function)
                # if "parser_variables" in sig.parameters:
                #     named_args["parser_variables"] = merged_variables
                if "parser_prefix" in sig.parameters:
                    named_args["parser_prefix"] = self.prefix
                if "parser" in sig.parameters:
                    named_args["parser"] = self
                if "partial_output" in sig.parameters:
                    named_args["partial_output"] = self._extend_prefix
                if "next_text" in sig.parameters:
                    if next_node is not None:
                        named_args["next_text"] = next_node.text
                    else:
                        named_args["next_text"] = ""
                if "prev_text" in sig.parameters:
                    if prev_node is not None:
                        named_args["prev_text"] = prev_node.text
                    else:
                        named_args["prev_text"] = ""

                command_output = command_function(*positional_args, **named_args)

                if "partial_output" not in sig.parameters:
                    self._extend_prefix(command_output)
            else:
                warnings.warn(f"Command '{command_name}' not found")
                command_output = ""
            return command_output

        elif node.expr_name == 'block_command_call':
            command_name, args = [self.visit(child) for child in node.children]
            return command_name, args

        elif node.expr_name == 'command_block_open':
            visited_children = [self.visit(child) for child in node.children]
            return visited_children[2]

        elif node.expr_name == 'command_block':

            # create a block content variable
            block_content = [node.children[1]]
            for child in node.children[2].children:
                if child.text == '':
                    continue
                block_content.append(child.children[0])
                block_content.append(child.children[1])
            self.block_content.append(block_content)

            start_block = self.visit(node.children[0])
            
            command_name, args = start_block
            if self.variable_exists(command_name):
                command_function = self.get_variable(command_name)
                positional_args = []
                named_args = {}
                for arg in args:
                    if isinstance(arg, PositionalArgument):
                        positional_args.append(arg.value)
                    elif isinstance(arg, NamedArgument):
                        named_args[arg.name] = arg.value
                sig = inspect.signature(command_function)
                # if "parser_variables" in sig.parameters:
                #     named_args["parser_variables"] = self.variable_stack[-1]
                if "parser_prefix" in sig.parameters:
                    named_args["parser_prefix"] = self.prefix
                if "parser" in sig.parameters:
                    named_args["parser"] = self
                if "block_content" in sig.parameters:
                    named_args["block_content"] = self.block_content[-1]
                if "partial_output" in sig.parameters:
                    named_args["partial_output"] = self._extend_prefix
                command_output = command_function(*positional_args, **named_args)
            else:
                command_output = ""

            node_text = node.text.replace("$", "DOLLAR_SIGN")
            self.block_content.pop()
            return f"__GMARKER_START_{command_name}${node_text}$___{command_output}__GMARKER_END_{command_name}$$___"
            # start_block(node.children[1], self)
            # end_block = self.visit(node.children[2])

        else:
            visited_children = []
            for i, child in enumerate(node.children):
                if len(node.children) > i + 1:
                    inner_next_node = node.children[i + 1]
                else:
                    inner_next_node = next_node
                if i > 0:
                    inner_prev_node = node.children[i - 1]
                else:
                    inner_prev_node = prev_node
                visited_children.append(self.visit(child, inner_next_node, inner_prev_node))
            # visited_children = [self.visit(child) for child in node.children]
            
            if len(visited_children) == 1:
                return visited_children[0]
            else:
                return "".join("" if c is None else c for c in visited_children)

    def get_variable(self, name, default_value=None):
        if name == "True":
            return True
        elif name == "False":
            return False
        
        parts = name.split(".")
        for variables in reversed(self.variable_stack):
            curr_pos = variables
            found = True
            for part in parts:
                if part in curr_pos:
                    curr_pos = curr_pos[part]
                else:
                    found = False
                    break
            if found:
                return curr_pos
        return default_value # variable not found

    def variable_exists(self, name):
        for var_dict in reversed(self.variable_stack):
            if name in var_dict:
                return True
        return False

    def set_variable(self, name, value):
        parts = name.split(".")
        found = True
        for variables in reversed(self.variable_stack):
            curr_pos = variables
            found = True
            for part in parts:
                if part in curr_pos:
                    if part == parts[-1]:
                        curr_pos[part] = value
                        break
                    else:
                        curr_pos = curr_pos[part]
                        if not isinstance(curr_pos, dict):
                            raise Exception(f"Cannot set variable '{name}' because '{part}' is not a dict")
                else:
                    if part == parts[-1] and len(parts) > 1: # setting a new property
                        curr_pos[part] = value
                    else:
                        found = False
                    break
            if found:
                break
        if not found:
            assert len(parts) == 1, "Can't set a property of a non-existing variable: " + name
            self.variable_stack[0][name] = value

    def _extend_prefix(self, text):
        prefix_out = strip_markers(str(text))
        self.prefix += prefix_out
        if self.prompt_object.echo:
            print(prefix_out, end='')
            sys.stdout.flush()
    
    def _trim_prefix(self, text):
        prefix_out = strip_markers(str(text))
        self.prefix = self.prefix[:-len(prefix_out)]
        # TODO: undo the echo if needed


def _generate(variable_name="generated", partial_output=None, parse=False, stop=None, max_tokens=500, n=1, echo=True, temperature=0.0, top_p=1.0, logprobs=None, parser_prefix=None, parser=None, prefix="", suffix="", next_text=None, prev_text=None, **kwargs):
    ''' Use the LM to generate a completion string that is stored in the variable `variable_name`.
    '''

    # if stop is None then we use the text of the node after the generate command
    if stop is None:
        if next_text is not None and prev_text is not None:

            # auto-detect quote stop tokens
            quote_types = ['"', "'", "'''", '"""', "`"]
            for quote_type in quote_types:
                if next_text.startswith(quote_type) and prev_text.endswith(quote_type):
                    stop = quote_type
                    break
                    
            # auto-detect XML tag stop tokens
            if stop is None:
                m = re.match(r"<([^>]+)>", next_text)
                if m is not None:
                    end_tag = "</"+m.group(1)+">"
                    if next_text.startswith(end_tag):
                        stop = end_tag
                else:
                    stop = next_text
                
        else:
            stop = next_text
    
    if temperature > 0:
        cache_seed = parser.prompt_object.cache_seed
        parser.prompt_object.cache_seed += 1
    else:
        cache_seed = 0

    gen_obj = parser.prompt_object.llm(
        parser_prefix+prefix, stop=stop, max_tokens=max_tokens, n=n,
        temperature=temperature, top_p=top_p, logprobs=parser.prompt_object.logprobs, cache_seed=cache_seed,
        echo=parser.prompt_object.logprobs is not None
    )
    if n == 1:
        generated_value = prefix+gen_obj["choices"][0]["text"]+suffix
        parser.set_variable(variable_name, generated_value)
        if logprobs is not None:
            parser.set_variable(variable_name+"_logprobs", gen_obj["choices"][0]["logprobs"])
    else:
        generated_values = [prefix+choice["text"]+suffix for choice in gen_obj["choices"]]
        parser.set_variable(variable_name, generated_values)
        if logprobs is not None:
            parser.set_variable(variable_name+"_logprobs", [choice["logprobs"] for choice in gen_obj["choices"]])

        # TODO: we could enable the parsing to branch into multiple paths here, but for now we just complete the prompt with the first prefix
        generated_value = generated_values[0]

    if parse:
        assert echo, "Cannot parse generated text if echo is disabled"
        subtree = grammar.parse(generated_value)
        return parser.visit(subtree)
    else:
        if n == 1:
            partial_output(generated_value)
            return generated_value
        else:

            # echoing with multipl completions is not standard behavior
            # this just uses the first generated value for completion and the rest as alternatives only used for the variable storage
            # we mostly support this so that the echo=False hiding behavior does not make multiple outputs more complicated than it needs to be in the UX
            if echo:
                partial_output(generated_value) 
            
            id = uuid.uuid4().hex
            l = len(generated_values)
            out = f"__GMARKER_generate_many_start_{echo}_{l}${id}$___"
            for i, value in enumerate(generated_values):
                if i > 0:
                    out += f"__GMARKER_generate_many_{echo}_{i}${id}$___"
                out += value
            return out + f"__GMARKER_generate_many_end${id}$___"
            # return "__GMARKER_generate_many_start$$___" + "__GMARKER_generate_many$$___".join([v for v in generated_values]) + "__GMARKER_generate_many_end$$___"
            # return "".join([v for v in generated_values])

def _add(*args):
    ''' Add the given variables together.
    '''
    return sum(args)

def _subtract(arg1, arg2):
    ''' Subtract the second variable from the first.
    '''
    return arg1 - arg2

def _strip(arg1):
    ''' Strip whitespace from the beginning and end of the given string.
    '''
    return arg1.strip()

def _each(list, block_content, parser, parser_prefix=None, stop=None, hidden=False, filter=None, batch_generate=False, batch_generate_temperature=0.0, batch_generate_max_tokens=500, batch_generate_top_p=1.0):
    ''' Iterate over a list and execute a block for each item.
    '''
    assert len(block_content) == 1
    assert not (hidden and batch_generate), "Cannot use hidden=True and batch_generate together"
    assert not (hidden and isinstance(list, str)), "Cannot use hidden=True and variable length iteration together" # TODO: perhaps we can relax this?
    echo = not hidden

    # an empty string means the variable was not provided
    if list is None:
        list = []

    # Make sure the list is iterable
    if filter is not None:

        # if the list is callable then we call it to get an iterable
        if callable(filter):
            # we support a set of optional arguments to the list function
            named_args = {}
            sig = inspect.signature(filter)
            if "template_context" in sig.parameters:
                named_args["template_context"] = {
                    "@block_text": block_content[0].text,
                    "@tokenizer": parser.prompt_object.tokenizer
                }
            list = filter(list, **named_args)
        else:
            raise TypeError("Can't apply a non-callable filter: " + str(filter))

    # make sure the list is iterable
    try:
        iter(list)
    except TypeError:
        raise TypeError("The #each command cannot iterate over a non-iterable value: " + str(list))

    out = []
    
    # if the list is a string then it is the name of a variable to save a new list to
    if isinstance(list, str):
        assert stop is not None, "Must provide a stop token when doing variable length iteration!"
        stop_tokens = [parser.prompt_object.llm.encode(s) for s in stop]

        if not batch_generate:
            i = 0
            data = []
            while True:
                parser.variable_stack.append({})
                parser.variable_stack[-1]["@index"] = i
                parser.variable_stack[-1]["@first"] = i == 0
                parser.variable_stack[-1]["this"] = {}
                out.append(parser.visit(block_content[0]))
                data.append(parser.variable_stack.pop()["this"])
                i += 1

                # we run a quick generation to see if we have reached the end of the list (not the +2 tokens is to help be tolorant to whitespace)
                gen_obj = parser.prompt_object.llm(parser.prefix, stop=stop, max_tokens=len(stop_tokens)+2, temperature=0, cache_seed=0)
                if gen_obj["choices"][0]["finish_reason"] == "stop":
                    break
        else:
            # create a pattern to match each item
            pattern = re.sub(
                r'{{generate [\'"]([^\'"]+)[\'"][^}]*}}',
                lambda x: r"(?P<"+x.group(1).replace("this.", "")+">.*?)",
                block_content[0].text
            )

            # generate the looped content
            if batch_generate_temperature > 0:
                cache_seed = parser.prompt_object.cache_seed
                parser.prompt_object.cache_seed += 1
            else:
                cache_seed = 0
            gen_obj = parser.prompt_object.llm(parser_prefix, stop=stop, max_tokens=batch_generate_max_tokens, temperature=batch_generate_temperature, top_p=batch_generate_top_p, cache_seed=cache_seed)
            generated_value = gen_obj["choices"][0]["text"]

            # parse the generated content (this assumes the generated content is syntactically correct)
            matches = re.finditer(pattern, generated_value)
            data = []
            for m in matches:#f"__GMARKER_START_{name}${node_text}$___{out}__GMARKER_END_{name}$$___"
                
                # get the variables that were generated
                match_dict = m.groupdict()
                if len(match_dict) == 1 and "this" in match_dict:
                    data.append(match_dict["this"])
                else:
                    data.append(match_dict)

                # recreate the output string with format markers added
                item_out = re.sub(
                    r"{{generate [\'\"]([^\'\"]+)[\'\"][^}]*}}",
                    lambda x: "__GMARKER_START_generate$"+x.group()+"$___"+match_dict[x.group(1).replace("this.", "")]+"__GMARKER_END_generate$$___",
                    block_content[0].text
                )
                out.append(item_out)
                
        parser.set_variable(list, data)

    # if the list is not a string then it is a list of items to iterate over
    else:
        parser.variable_stack.append({})
        for i, item in enumerate(list):
            parser.variable_stack[-1]["@index"] = i
            parser.variable_stack[-1]["@first"] = i == 0
            parser.variable_stack[-1]["@last"] = i == len(list) - 1
            parser.variable_stack[-1]["this"] = item
            item_out = parser.visit(block_content[0])
            if not echo:
                parser._trim_prefix(item_out)
            out.append(item_out)
        parser.variable_stack.pop()
    if echo:
        return "__GMARKER_each$$___" + "__GMARKER_each$$___".join(out) + "__GMARKER_each$$___"    
    else:
        id = uuid.uuid4().hex
        l = len(out)
        out_str = f"__GMARKER_each_noecho_start_{echo}_{l}${id}$___"
        for i, value in enumerate(out):
            if i > 0:
                out_str += f"__GMARKER_each_noecho_{echo}_{i}${id}$___"
            out_str += value
        return out_str + f"__GMARKER_each_noecho_end${id}$___"
        # return "__GMARKER_each_noecho$$___" + "__GMARKER_each_noecho$$___".join(out) + "__GMARKER_each_noecho$$___"
        

def _select(variable_name="selected", block_content=None, parser=None, partial_output=None, parser_prefix=None, logprobs=None):
    ''' Select a value from a list of choices.
    '''
    assert len(block_content) > 1
    options = [block_content[0].text]
    for i in range(1, len(block_content), 2):
        assert block_content[i].text == "{{or}}"
        options.append(block_content[i+1].text)

    option_tokens = [parser.prompt_object.llm.encode(option) for option in options]

    option_logprobs = {}
    for option in option_tokens:
        option_logprobs[parser.prompt_object.llm.decode(option)] = 0

    # [TODO] we should force the LM to generate a valid specific option
    #        for openai this means setting logprobs to valid token ids
    gen_obj = parser.prompt_object.llm(
        parser_prefix,
        max_tokens=max([len(o) for o in option_tokens]),
        logprobs=10,
        cache_seed=0
    )

    # compute logprobs for each option
    top_logprobs = gen_obj["choices"][0]["logprobs"]["top_logprobs"]
    for i in range(len(top_logprobs)):
        for option in option_tokens:
            if len(option) > i:
                option_logprobs[parser.prompt_object.llm.decode(option)] += top_logprobs[i].get(option[i], -100)
    
    # penalize options that are too long
    for option in option_tokens:
        if len(option) > len(top_logprobs):
            option_logprobs[parser.prompt_object.llm.decode(option)] -= 100

    selected_option = max(option_logprobs, key=option_logprobs.get)
    parser.set_variable(variable_name, selected_option)
    if logprobs is not None:
        parser.set_variable(logprobs, option_logprobs)
    
    if max(option_logprobs.values()) <= -100:
        raise ValueError("No valid option generated in #select, this could be fixed if we used a tokenizer and forced the LM to use a valid option! The top logprobs were" + str(top_logprobs))
    
    partial_output(selected_option)

    return selected_option

def _if(value, block_content, parser, reverse=False):
    ''' Standard if/else statement.
    '''
    assert len(block_content) in [1,2] # we don't support else if yet...
    options = [block_content[0]]
    for i in range(1, len(block_content), 2):
        assert block_content[i].text == "{{else}}"
        options.append(block_content[i+1])

    if isinstance(value, str):
        value = value.lower().strip() in ["true", "yes", "1", "on", "t", "y", "ok", "okay"]
    
    if reverse:
        value = not value
    
    if value:
        return parser.visit(options[0])
    elif len(options) > 1:
        return parser.visit(options[1])
    else:
        return ""


def _unless(value, block_content, parser):
    return _if(value, block_content, parser, reverse=True)

def _block(name=None, block_content=None, parser=None, hidden=False):
    ''' Generic block definition.
    '''
    assert parser is not None
    
    out = parser.visit(block_content[0])
    if name is not None:
        parser.set_variable(name, strip_markers(out))
    if hidden:
        parser._trim_prefix(out)
    
    return out

def _set(name, value=None, parser=None):
    ''' Set the value of a variable or set of variables.

    Parameters
    ----------
    name : str or dict
        If a string, the name of the variable to set. If a dict, the keys are the variable names and the values are the values to set.
    value : str, optional
        The value to set the variable to. Only used if `name` is a string.
    '''
    assert parser is not None

    if isinstance(name, dict):
        for k, v in name.items():
            parser.set_variable(k, v)
        out = ""
        for k, v in name.items():
            if isinstance(v, str):
                if "\n" in v:
                    v = f'"""{v}"""'
                elif '"' in v:
                    v = f"'{v}'"
                else:
                    v = f'"{v}"'
            out += f" {k}={v}"
        out += ""
        return f"__GMARKER_set${out}$___"
    else:
        parser.set_variable(name, value)
        out = "{{set "+name+"=" + value + "}}"
        return f"__GMARKER_set${out}$___"
