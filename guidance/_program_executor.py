import traceback
import ast
import inspect
import re
import asyncio
import logging
import pyparsing as pp
from ._grammar import grammar
from ._variable_stack import VariableStack
log = logging.getLogger(__name__)


class ProgramExecutor():
    def __init__(self, program):
        """ Attaches this executor to a program object.
        """

        self.program = program
        self.block_content = []
        self.executing = True
        self.should_stop = False
        self.caught_stop_iteration = False
        self.llm_session = None
        self._logging = hasattr(self.program.log, "append")

        # find all the handlebars-style partial inclusion tags and replace them with the partial template
        # def replace_partial(match):
        #     parts = match.group(1).split(" ", 1)
        #     partial_name = parts[0]
            
        #     # ,args_string = match.group(1).split(" ", 1)
        #     if partial_name not in program._variables:
        #         raise ValueError("Partial '%s' not given in the keyword args:" % partial_name)
        #     out = "{{#block '"+partial_name+"'"
        #     if len(parts) > 1:
        #         out += " " + parts[1]
        #     out += "}}" + program._variables[partial_name].text + "{{/block}}"
        #     # Update the current program variables using those from the partial, but do not overwrite.
        #     # (Rebuilding the _variables map here would break returning new values to the program variables later, e.g. from gen.)
        #     update_variables = {
        #         k: v
        #         for k, v in program[partial_name]._variables.items()
        #         if k not in program._variables
        #     }
        #     program._variables.update(update_variables)
        #     return out
        # text = re.sub(r"{{>(.*?)}}", replace_partial, program._text)

        # parse the program text
        try:
            self.parse_tree = grammar.parse_string(program._text)
        except (pp.ParseException, pp.ParseSyntaxException) as e:
            initial_str = program._text[max(0, e.loc-40):e.loc]
            initial_str = initial_str.split("\n")[-1] # trim off any lines before the error
            next_str = program._text[e.loc:e.loc+40]
            error_string = str(e)
            if next_str.startswith("{{#") or next_str.startswith("{{~#"):
                error_string += "\nPerhaps the block command was not correctly closed?"
            msg = error_string + "\n\n"+initial_str
            # msg += "\033[91m" + program._text[e.loc:e.loc+40] + "\033[0m\n"
            msg += program._text[e.loc:e.loc+40] + "\n"
            msg += " " * len(initial_str) + "^\n"

            raise SyntaxException(msg, e) from None
            
    # def _check_for_simple_error(self, text):
    #     """ Check for a simple errors in the program text, and give nice error messages.
    #     """

    #     vars = self.program._variables

    #     # missing block pound sign
    #     for k in vars:
    #         if getattr(vars[k], "is_block", False):

    #             # look for block commands that are missing the opening pound sign or closing slash
    #             m = re.search(r"(^|[^\\]){{\s*"+k+"(\s|}|~)", text)
    #             if m is not None:
    #                 # get the context around the matching error
    #                 start = max(0, m.start()-30)
    #                 end = min(len(text), m.end()+30)
    #                 context = text[start:end]
    #                 if start > 0:
    #                     context = "..."+context
    #                 if end < len(text):
    #                     context = context+"..."
    #                 raise ValueError("The guidance program is missing the opening pound (#) sign or closing slash (/) for the block level command `"+k+"` at:\n"+context) from None
                
    #             # look for block commands that are missing the closing tag
    #             num_opens = len(re.findall(r"(^|[^\\]){{~?#\s*"+k+"(\s|}|~)", text))
    #             num_closes = len(re.findall(r"(^|[^\\]){{~?/\s*"+k+"(\s|}|~)", text))
    #             if num_opens > num_closes:
    #                 raise ValueError("The guidance program is missing a closing tag for the block level command `"+k+"`.") from None
    #             if num_opens < num_closes:
    #                 raise ValueError("The guidance program is missing an opening tag for the block level command `"+k+"`.") from None
        
        

    async def run(self, llm_session):
        """ Execute the program.
        """
        self.llm_session = llm_session
        try:
            # first parse all the whitespace control
            # self.whitespace_control_visit(self.parse_tree)

            # now execute the program
            self.program._variables["@raw_prefix"] = ""
            await self.visit(self.parse_tree, VariableStack([self.program._variables], self))
        except Exception as e:
            print(traceback.format_exc())
            print("Error in program: ", e)
            raise e
        
    def stop(self):
        self.should_stop = True

    # def process_content(self, text):
    #     if text.endswith("{{!--GSTRIP--}}"):
    #         text = text[:-15].rstrip()
    #     if text.startswith("{{!--GSTRIP--}}"):
    #         text = text[15:].lstrip()
    #     return text

    # def whitespace_control_visit(self, node, next_node=None, prev_node=None, parent_node=None, grandparent_node=None):
    #     if node_name in ('command', 'block_command_open', 'block_command_sep', 'block_command_close'):
    #         if node.text.startswith("{{~"):
    #             if prev_node and prev_node_name == "content":
    #                 prev_node.text = prev_node.text + "{{!--GSTRIP--}}"
    #         if node.text.endswith("~}}"):
    #             if next_node and next_node_name == "content":
    #                 next_node.text = "{{!--GSTRIP--}}" + next_node.text

    #     # visit all our children
    #     for i, child in enumerate(node.children):
    #         if len(node.children) > i + 1:
    #             inner_next_node = node.children[i + 1]
    #         else:
    #             inner_next_node = next_node
    #         if i > 0:
    #             inner_prev_node = node.children[i - 1]
    #         else:
    #             inner_prev_node = prev_node
    #         self.whitespace_control_visit(child, inner_next_node, inner_prev_node, node, parent_node)
    
    async def visit(self, node, variable_stack, next_node=None, next_next_node=None, prev_node=None, parent_node=None, grandparent_node=None):

        # if we are after a break point then we return nothing
        # (note that this flag will be cleared once the loop is ended)
        if self.caught_stop_iteration:
            return ""
        
        node_name = node.get_name()

        if node_name == 'variable_name':
            return node.text

        elif node_name == 'content':
            variable_stack["@raw_prefix"] += node[0]
            return None
        
        elif node_name == 'long_comment':
            if node.text.startswith("{{!--G"):
                variable_stack["@raw_prefix"] += node.text
            return ""
        
        elif node_name == 'comment':
            return ""
        
        elif node_name == 'partial':
            partial_program = variable_stack[node[0]["name"]]
            tree = grammar.parse_string(partial_program._text)
            partial_args = [await self.visit(child, variable_stack) for child in node["command_call"][1:]]
            args = []
            kwargs = {}
            for arg in partial_args:
                if isinstance(arg, PositionalArgument):
                    args.append(arg.value)
                elif isinstance(arg, NamedArgument):
                    kwargs[arg.name] = arg.value
            partial_vars = {k: v for k,v in partial_program.variables().items() if k not in ["llm", "logging"] and k not in variable_stack}
            if len(args) > 0:
                partial_vars["args"] = args
            partial_vars.update(kwargs)
            variable_stack.push(partial_vars)
            out = await self.visit(tree, variable_stack)
            variable_stack.pop()
            return out

        elif node_name == 'positional_command_arg':
            return PositionalArgument(await self.visit(node[0], variable_stack))

        elif node_name == 'named_command_arg':
            return NamedArgument(node[0], await self.visit(node[2], variable_stack))

        elif node_name == 'command_name':
            return node.text
        
        elif node_name == 'command_name':
            return node.text

        elif node_name == 'escaped_command':
            variable_stack["@raw_prefix"] += node.text[1:]
            return
        
        elif node_name == 'boolean_literal':
            if node[0] == "True":
                return True
            elif node[0] == "False":
                return False
            else:
                raise Exception("Invalid boolean literal")

        elif node_name == 'number_literal':
            if "." in node[0]:
                return float(node[0])
            else:
                return int(node[0])
            
        elif node_name == 'string_literal':
            return node[0]
        
        elif node_name == 'object_literal':
            out = {}
            for i in range(0, len(node), 2):
                key = await self.visit(node[i], variable_stack)
                value = await self.visit(node[i + 1], variable_stack)
                out[key] = value
            return out
        
        elif node_name == 'array_literal':
            return [await self.visit(node[i], variable_stack) for i in range(0, len(node))]

        elif node_name == 'literal':
            try:
                return ast.literal_eval(node.text)
            except Exception as e:
                raise Exception(f"Error parsing literal: {node.text} ({e})")

        elif node_name == 'command':

            # if execution is already stopped before we start the command we just keep the command text
            if not self.executing:
                variable_stack["@raw_prefix"] += node.text
                return
            
            # mark our position in case we need to rewind
            # pos = len(self.prefix)

            # find the command name
            if "variable_ref" in node:
                name = "variable_ref"
            elif "keyword" in node:
                name = "keyword"
            elif "command_call" in node:
                name = node["command_call"]["name"]
            else: # binary_operator and unary_operator
                name = node[0].get_name()

            # add the start marker
            escaped_node_text = node.text.replace("$", "&#36;").replace("{", "&#123;").replace("}", "&#125;")
            variable_stack["@raw_prefix"] += "{{!--"+f"GMARKER_START_{name}${escaped_node_text}$"+"--}}"
            
            # visit our children
            self.block_content.append([])
            visited_children = [await self.visit(child, variable_stack, next_node, next_next_node, prev_node, node, parent_node) for child in node]
            self.block_content.pop()
            out = "".join("" if c is None else str(c) for c in visited_children)

            variable_stack["@raw_prefix"] += out +  "{{!--" + f"GMARKER_END_{name}$$" + "--}}"

            # if execution became stopped during the command, we append the command text
            if not self.executing:
                # self.reset_prefix(pos)
                variable_stack["@raw_prefix"] += node.text
            return

        elif node_name == 'paren_group':
            visited_children = [await self.visit(child, variable_stack) for child in node.children]
            return visited_children[1]

        elif node_name == 'command_call' or node_name == 'variable_ref' or node_name == 'binary_operator' or node_name == 'unary_operator' or node_name == 'keyword':
            if node_name == 'command_call':
                command_name = node["name"]
                args = [await self.visit(child, variable_stack) for child in node[1:]]
            elif node_name == 'binary_operator':
                command_name = "BINARY_OPERATOR_" + node["operator"]
                args = [
                    PositionalArgument(await self.visit(node["lhs"], variable_stack)),
                    PositionalArgument(await self.visit(node["rhs"], variable_stack))
                ]
            elif node_name == 'unary_operator':
                command_name = "UNARY_OPERATOR_" + node["operator"]
                args = [PositionalArgument(await self.visit(node["value"], variable_stack))]
            else:
                command_name = node[0]
                args = []

            # if the command arguments stopped execution, we don't execute the command
            if not self.executing:
                return
            
            # return_value = ""
            if command_name in variable_stack:
                command_function = variable_stack[command_name]

                # we convert a variable reference to a function that returns the variable value
                if node_name == "variable_ref":
                    command_value = command_function
                    command_function = lambda: command_value

                # check for a generated call statement
                named_args = {}
                if isinstance(command_function, str):
                    call_details = variable_stack["extract_function_call"](command_function)
                    if call_details is None:
                        raise Exception(f"Can't call the string (there is no function call recognized by `extract_function_call` in it): {command_function}")
                    
                    command_function = call_details.__name__
                    if command_function not in variable_stack:
                        raise Exception(f"Function {command_function} not found!")
                    else:
                        command_function = variable_stack[command_function]
                        named_args = call_details.__kwdefaults__


                # def update_return_value(s):
                #     nonlocal return_value
                #     if return_value == "":
                #         return_value = s
                    
                #     # convert to strings if we are concatenating
                #     else:
                #         return_value += "" if s is None else str(s)

                # If we are a top level command we extend the prefix
                top_level = parent_node is not None and parent_node.get_name() == "command"
                    # partial_output = self.extend_prefix
                    # pass
                
                # otherwise we keep track of output locally so we can return it
                if not top_level:
                    # partial_output = update_return_value
                    pos = len(variable_stack["@raw_prefix"])
                    variable_stack.push({"@raw_prefix": variable_stack["@raw_prefix"], "@no_display": True})

                # create the arguments for the command
                positional_args = []
                for arg in args:
                    if isinstance(arg, PositionalArgument):
                        positional_args.append(arg.value)
                    elif isinstance(arg, NamedArgument):
                        named_args[arg.name] = arg.value
                sig = inspect.signature(command_function)
                if "_parser_context" in sig.parameters:
                    named_args["_parser_context"] = {
                        "parser": self,
                        "variable_stack": variable_stack,
                        "next_node": next_node,
                        "next_next_node": next_next_node,
                        "prev_node": prev_node,
                        "block_content": None
                    }

                # call the command
                if self._logging:
                    self.program.log.append({
                        "type": "start",
                        "name": command_name,
                        "positional_args": positional_args,
                        "named_args": {k:v for k,v in named_args.items() if k != "_parser_context"},
                        "@prefix": variable_stack["@prefix"],
                        # "node_id": id(node)
                    })
                    pos = len(variable_stack["@prefix"])
                try:
                    if inspect.iscoroutinefunction(command_function):
                        await asyncio.sleep(0) # give other coroutines a chance to run
                        command_output = await command_function(*positional_args, **named_args)
                    else:
                        command_output = command_function(*positional_args, **named_args)
                except StopIteration as ret:
                    command_output = ret.value
                    self.caught_stop_iteration = True
                if self._logging:
                    self.program.log.append({"type": "end", "name": command_name, "new_prefix": variable_stack["@prefix"][pos:]})

                # call partial output if the command didn't itself (and we are still executing)
                if not top_level:
                    curr_prefix = variable_stack.pop()["@raw_prefix"] # pop the variable stack we pushed earlier becuause we were hidden
                    if command_output is not None:
                        return command_output
                    else:
                        new_content = curr_prefix[pos:]

                        # see if we got a list of outputs encoded as a string
                        parts = re.split(r"{{!--GMARKERmany[^}]+}}", new_content)
                        if len(parts) > 1:
                            return parts[1:-1]
                        else:
                            return new_content
                else:
                    if command_output is not None:
                        variable_stack["@raw_prefix"] += str(command_output)
                    return ""
            else:
                # if the variable does not exist we just pause execution
                if self.program.await_missing:
                    self.executing = False
                    return None
                else:
                    # raise an error if the command doesn't exist
                    raise KeyError("Command/variable '"+command_name+"' not found! Please pass it when calling the program (or set a default value for it when creating the program).")
            
            # # if we are not a top level command we return the output instead of displaying it
            # if not top_level:
            #     return_value = variable_stack.pop()["@raw_prefix"][pos:]

            #     # see if we got a list of outputs encoded as a string
            #     parts = re.split(r"{{!--GMARKERmany[^}]+}}", return_value)
            #     if len(parts) > 1:
            #         return parts[1:-1]
            #     else:
            #         return return_value
            # else:
            #     return ""

        elif node_name == 'block_command_call':
            parts = [await self.visit(child, variable_stack) for child in node.children]
            if len(parts) > 1:
                command_name, args = parts
            else:
                command_name = parts[0]
                args = []
            return command_name, args

        elif node_name == 'block_command_open':
            return await self.visit(node.children[2], variable_stack)
            # visited_children = [await self.visit(child) for child in node.children]
            # return visited_children[2]

        elif node_name == 'block_command':

            # if execution is already stopped before we start the command block we just return unchanged
            if not self.executing:
                variable_stack["@raw_prefix"] += node.text
                return ""

            # create a block content variable
            # block_content = node["block_content"]
            # block_content = [node.children[1]]
            # for child in node.children[2].children:
            #     if child.text == '':
            #         continue
            #     block_content.append(child.children[0])
            #     block_content.append(child.children[1])
            assert node[1].get_name() == "block_content" # TODO: figure out why node["block_content"] doesn't work (has to do with SavedText messing up the keys)
            self.block_content.append(node[1])

            # get the command name and arguments
            call = node["command_call"]
            command_name = call["name"]
            command_args = [await self.visit(arg, variable_stack) for arg in call[1:]]
            
            # command_args = []
            # if "positional_command_arg" in call:
            #     for arg in call["positional_command_arg"]:
            #         command_args.append(await self.visit(arg, variable_stack))
            # if "named_command_arg" in call:
            #     for arg in call["named_command_arg"]:
            #         command_args.append(await self.visit(arg, variable_stack))
            # command_name, command_args = [await self.visit(arg, variable_stack) for arg in node["command_call"]["args"]]

            # make sure we have a matching end command TODO: move this to a parser action
            # if not (node.text.endswith("/"+command_name+"}}") or node.text.endswith("/"+command_name+"~}}")):
            #     raise SyntaxError("Guidance command block starting with `"+node.text[:20]+"...` does not end with a matching `{{/"+command_name+"}}` but instead ends with `..."+node.text[-20:]+"!")

            # if execution stops while parsing the start command just return unchanged
            if not self.executing:
                variable_stack["@raw_prefix"] += node.text
                return ""

            # add the start marker
            escaped_node_text = node.text.replace("$", "&#36;").replace("{", "&#123;").replace("}", "&#125;")
            start_marker = "{{!--"+f"GMARKER_START_{command_name}${escaped_node_text}$"+"--}}"
            variable_stack["@raw_prefix"] += start_marker

            if command_name in variable_stack:
                command_function = variable_stack[command_name]
                positional_args = []
                named_args = {}
                for arg in command_args:
                    if isinstance(arg, PositionalArgument):
                        positional_args.append(arg.value)
                    elif isinstance(arg, NamedArgument):
                        named_args[arg.name] = arg.value
                
                # see if the command expects parser context
                sig = inspect.signature(command_function)
                if "_parser_context" in sig.parameters:
                    named_args["_parser_context"] = {
                        "parser": self,
                        "block_content": self.block_content[-1],
                        "variable_stack": variable_stack,
                        "parser_node": node,
                        "block_close_node": node[-1],
                        "next_node": next_node,
                        "next_next_node": next_next_node,
                        "prev_node": node[0]
                    }
                
                # call the optionally asyncronous command
                if self._logging:
                    self.program.log.append({
                        "type": "start",
                        "name": command_name,
                        "positional_args": positional_args,
                        "named_args": {k:v for k,v in named_args.items() if k != "_parser_context"},
                        "@prefix": variable_stack["@prefix"],
                        # "node_id": id(node)
                    })
                    pos = len(variable_stack["@prefix"])
                if inspect.iscoroutinefunction(command_function):
                    command_output = await command_function(*positional_args, **named_args)
                else:
                    command_output = command_function(*positional_args, **named_args)
                if self._logging:
                    self.program.log.append({"type": "end", "name": command_name, "new_prefix": variable_stack["@prefix"][pos:]})

                # if the command didn't send partial output we do it here
                if command_output is not None:
                    variable_stack["@raw_prefix"] += command_output

            # pop off the block content after the command call
            self.block_content.pop()

            variable_stack["@raw_prefix"] += "{{!--" + f"GMARKER_END_{command_name}$$" + "--}}"
            return

        else:
            visited_children = []
            for i, child in enumerate(node):
                if len(node) > i + 1:
                    inner_next_node = node[i + 1]
                else:
                    inner_next_node = next_node
                
                if len(node) > i + 2:
                    inner_next_next_node = node[i + 2]
                elif len(node) == i + 2:
                    inner_next_next_node = next_node
                else:
                    inner_next_next_node = next_next_node
                
                if i > 0:
                    inner_prev_node = node[i - 1]
                else:
                    inner_prev_node = prev_node
                visited_children.append(await self.visit(child, variable_stack, inner_next_node, inner_next_next_node, inner_prev_node, node, parent_node))
            # visited_children = [self.visit(child) for child in node.children]
            
            if len(visited_children) == 1:
                return visited_children[0]
            else:
                return "".join("" if c is None else c for c in visited_children)

    # def get_variable(self, name, default_value=None):
    #     parts = re.split(r"\.|\[", name) 40 ms 2048 12B
    #     for variables in reversed(self.variable_stack):
    #         curr_pos = variables
    #         found = True
    #         for part in parts:
    #             if part.endswith("]"):
    #                 var_part = ast.literal_eval(part[:-1])
    #             else:
    #                 var_part = part
    #             try:
    #                 next_pos = curr_pos[var_part]
    #                 next_found = True
    #             except KeyError:
    #                 next_found = False
    #             if next_found:
    #                 curr_pos = next_pos
    #             else:
    #                 found = False
    #                 break
    #         if found:
    #             return curr_pos
    #     return default_value # variable not found

    # def variable_exists(self, name):
    #     return self.get_variable(name, _NO_VALUE) != _NO_VALUE

    # def set_variable(self, name, value):
    #     parts = re.split(r"\.|\[", name)
    #     # parts = name.split(".")
    #     found = True
    #     for variables in reversed(self.variable_stack):
    #         curr_pos = variables
    #         found = True
    #         for part in parts:
    #             if part.endswith("]"):
    #                 var_part = ast.literal_eval(part[:-1])
    #             else:
    #                 var_part = part
    #             try:
    #                 next_pos = curr_pos[var_part]
    #                 next_found = True
    #             except KeyError:
    #                 next_found = False
                
    #             if next_found:
    #                 if part == parts[-1]:
    #                     curr_pos[var_part] = value
    #                     break
    #                 else:
    #                     curr_pos = next_pos
    #             else:
    #                 if part == parts[-1] and len(parts) > 1: # setting a new property
    #                     curr_pos[var_part] = value
    #                 else:
    #                     found = False
    #                 break
    #         if found:
    #             break
    #     if not found:
    #         assert len(parts) == 1, "Can't set a property of a non-existing variable: " + name
    #         self.variable_stack[0][name] = value

    # def extend_prefix(self, text, variable_stack):
    #     if text == "" or text is None:
    #         return
    #     variable_stack["@raw_prefix"] += str(text)
    #     self.program.update_display()
    
    # def reset_prefix(self, pos):
    #     self.prefix = self.prefix[:pos]
    #     self.program.update_display()
    #     # TODO: undo the echo if needed

class PositionalArgument:
    def __init__(self, value):
        self.value = value

class NamedArgument:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class StopCompletion(Exception):
    pass

class SyntaxException(Exception):
    def __init__(self, msg, pyparsing_exception=None):
        super().__init__(msg)
        self.pyparsing_exception = pyparsing_exception