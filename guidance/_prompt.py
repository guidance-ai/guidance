import ast
import inspect
import re
import html
import sys
import parsimonious
import warnings
from . import generators
import guidance

class PromptCompletion:
    ''' Represents the result of an executed prompt.
    '''
    def __init__(self, variables, completed_text, completed_text_html, prompt):
        self.variables = variables
        self.completed_text = completed_text
        self.completed_text_html = completed_text_html
        self.prompt = prompt

    def __getitem__(self, key):
        return self.variables[key]

    def __repr__(self):
        return self.completed_text

    def _repr_html_(self):
        return self.completed_text_html

class Prompt:
    ''' A prompt template that can be compiled and executed to generate a PromptCompletion result.
    '''

    def __init__(self, prompt, generator=None, echo=False):
        """ Create a new Prompt object from a prompt string.
        """
        self.prompt_string = prompt
        self.generator = generator
        self.echo = echo
        self.tree = grammar.parse(self.prompt_string)

        # default to an OpenAI generator
        if self.generator is None:
            self.generator = guidance.default_generator
    
    def __call__(self, **kwargs):
        built_ins = {
            "generate": _generate,
            "each": _each,
            "select": _select,
            "if": _if,
            "unless": _unless,
        }
        variables = {}
        variables.update(built_ins)
        variables.update(kwargs)

        vi = TopDownVisitor(variables, self)

        output = vi.visit(self.tree)
        
        # remove the built-ins from the variables we return
        for k in built_ins:
            del variables[k]

        display_out = html.escape(output)
        display_out = re.sub(r"__GMARKER_START_generate\$([^\$]*)\$___", r"<span style='background-color: rgb(0, 165, 0, 0.25); display: inline;' title='\1'>", display_out)
        display_out = display_out.replace("__GMARKER_END_generate$___", "</span>")
        display_out = re.sub(r"__GMARKER_START_select\$([^\$]*)\$___", r"<span style='background-color: rgb(0, 165, 0, 0.25); display: inline;' title='\1'>", display_out)
        display_out = display_out.replace("__GMARKER_END_select$___", "</span>")
        display_out = re.sub(r"__GMARKER_START_variable_ref\$([^\$]*)\$___", r"<span style='background-color: rgb(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='\1'>", display_out)
        display_out = display_out.replace("__GMARKER_END_variable_ref$___", "</span>")
        display_out = display_out.replace("__GMARKER_each$___", "<div style='border-left: 1px dashed rgb(0, 0, 0, .2); border-top: 0px solid rgb(0, 0, 0, .2); margin-right: -4px; display: inline; width: 4px; height: 24px;'></div>")
        display_out = re.sub(r"__GMARKER_START_([^\$]*)\$([^\$]*)\$___", r"<span style='background-color: rgb(0, 165, 0, 0.25); display: inline;' title='\2'>", display_out)
        display_out = re.sub(r"__GMARKER_END_([^\$]*)\$___", "</span>", display_out)
        display_out = "<pre style='padding: 7px; border-radius: 4px; background: white; white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 16px; line-height: 24px; color: #000'>"+display_out+"</pre>"

        # strip out the markers for the unformatted output
        output = re.sub(r"__GMARKER_([^\$]*)\$___", r"", output)
        output = output.replace("__GMARKER_([^_]+)_END___", "")

        return PromptCompletion(variables, output, display_out, self)


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
variable_ref = !"or" !"else" ~r"[a-z][a-z_0-9\.]*"i
variable_name = ~r"[a-z][a-z_0-9\.]*"i
contentw = ~r'.*'
content = ~r"[^{\\]*"
unrelated_escape = "\\" !command_start
escaped_command = "\\" command_start ~r"[^}]*" command_end
literal = ~r'"[^\"]*"' / ~r"'[^\']*'" / ~r"[0-9\.]+"
""")

class PositionalArgument:
    def __init__(self, value):
        self.value = value

class NamedArgument:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class TopDownVisitor():
    def __init__(self, variables, prompt_object):
        self.prefix = ''
        self.variable_stack = [variables]
        self.prompt_object = prompt_object
    
    def visit(self, node):

        if node.expr_name == 'variable_ref':
            return self.get_variable(node.text)

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
            visited_children = [str(self.visit(child)) for child in node.children]
            out = "".join(visited_children) or ""
            
            command_head = node.children[1].children[0]
            if command_head.expr_name == 'variable_ref':
                self._extend_prefix(out)
                name = "variable_ref"
            elif command_head.expr_name == 'command_call':
                name = command_head.children[0].text
            else:
                raise Exception("Unknown command head type: "+command_head.expr_name)

            return f"__GMARKER_START_{name}${node.text}$___{out}__GMARKER_END_{name}$___"

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
                    block_content = [node.children[1]]
                    for child in node.children[2:-1]:
                        if child.text == '':
                            continue
                        block_content.append(child.children[0].children[0])
                        block_content.append(child.children[0].children[1])
                    named_args["block_content"] = block_content
                if "partial_output" in sig.parameters:
                    named_args["partial_output"] = self._extend_prefix
                command_output = command_function(*positional_args, **named_args)
            else:
                command_output = ""
            return f"__GMARKER_START_{command_name}${node.text}$___{command_output}__GMARKER_END_{command_name}$___"
            # start_block(node.children[1], self)
            # end_block = self.visit(node.children[2])

        else:
            visited_children = [self.visit(child) for child in node.children]
            
            if len(visited_children) == 1:
                return visited_children[0]
            else:
                return "".join(visited_children) or ""

    def get_variable(self, name, default_value=""):
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
            assert len(parts) == 1, "Can't set a property of a non-existing variable"
            self.variable_stack[0][name] = value

    def _extend_prefix(self, text):
        prefix_out = re.sub(r"__GMARKER_START_([^\$]*)\$([^\$]*)\$___", "", text)
        prefix_out = re.sub(r"__GMARKER_END_([^\$]*)\$___", "", prefix_out)
        self.prefix += prefix_out
        if self.prompt_object.echo:
            print(prefix_out, end='')
            sys.stdout.flush()


def _generate(variable_name, partial_output, stop=None, max_tokens=500, parser_prefix=None, parser=None, prefix="", suffix=""):
    ''' Use the LM to generate a completion string that is stored in the variable `variable_name`.
    '''
    gen_obj = parser.prompt_object.generator(parser_prefix+prefix, stop=stop, max_tokens=max_tokens)
    generated_value = prefix+gen_obj["choices"][0]["text"]+suffix
    parser.set_variable(variable_name, generated_value)
    subtree = grammar.parse(generated_value)
    parsed_text = parser.visit(subtree)
    return parsed_text

def _each(list, block_content, parser):
    ''' Iterate over a list and execute a block for each item.
    '''
    assert len(block_content) == 1
    out = []
    parser.variable_stack.append({})
    for i, item in enumerate(list):
        parser.variable_stack[-1]["@index"] = i
        parser.variable_stack[-1]["@first"] = i == 0
        parser.variable_stack[-1]["@last"] = i == len(list) - 1
        parser.variable_stack[-1]["this"] = item
        out.append(parser.visit(block_content[0]))
    parser.variable_stack.pop()
    return "__GMARKER_each$___" + "__GMARKER_each$___".join(out) + "__GMARKER_each$___"

def _select(variable_name, block_content, parser, partial_output, parser_prefix=None, logprobs=None):
    ''' Select a value from a list of choices.
    '''
    assert len(block_content) > 1
    options = [block_content[0].text]
    for i in range(1, len(block_content), 2):
        assert block_content[i].text == "{{or}}"
        options.append(block_content[i+1].text)

    option_tokens = parser.prompt_object.generator.tokenize(options)

    option_logprobs = {}
    for option in option_tokens:
        option_logprobs["".join(option)] = 0

    # [TODO] we should force the LM to generate a valid specific option
    #        for openai this means setting logprobs to valid token ids
    gen_obj = parser.prompt_object.generator(
        parser_prefix,
        max_tokens=max([len(o) for o in option_tokens]),
        logprobs=10
    )

    # copmute logprobs for each option
    top_logprobs = gen_obj["choices"][0]["logprobs"]["top_logprobs"]
    for i in range(len(top_logprobs)):
        for option in option_tokens:
            if len(option) > i:
                option_logprobs["".join(option)] += top_logprobs[i].get(option[i], -100)
    
    # penalize options that are too long
    for option in option_tokens:
        if len(option) > len(top_logprobs):
            option_logprobs["".join(option)] -= 100

    selected_option = max(option_logprobs, key=option_logprobs.get)
    parser.set_variable(variable_name, selected_option)
    if logprobs is not None:
        parser.set_variable(logprobs, option_logprobs)
    
    if max(option_logprobs.values()) <= -100:
        raise ValueError("No valid option generated in #select, this could be fixed if we used a tokenizer and forced the LM to use a valid option!")
    
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
