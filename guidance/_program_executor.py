import traceback
import ast
import inspect
import re
import asyncio
import warnings
import parsimonious
from ._utils import strip_markers
from ._grammar import grammar

class ProgramExecutor():
    def __init__(self, program):
        """ Attaches this executor to a program object.
        """

        self.variable_stack = [program.variables]
        self.program = program
        self.prefix = ""
        # self.prefix_tokens = []
        self.block_content = []
        self.executing = True
        self.should_stop = False
        self.caught_stop_iteration = False

        # parse the program text
        self.parse_tree = grammar.parse(program._text)

    async def run(self):
        """ Execute the program.
        """
        try:
            await self.visit(self.parse_tree)
        except Exception as e:
            print(traceback.format_exc())
            print("Error in program: ", e)
            raise e
        
    def stop(self):
        self.should_stop = True
    
    async def visit(self, node, next_node=None, prev_node=None, parent_node=None, grandparent_node=None):

        if node.expr_name == 'variable_name':
            return node.text

        elif node.expr_name == 'content':

            # check for white space stripping commands
            if next_node is not None and next_node.text.startswith("{{~"):
                text = node.text.rstrip()
            elif prev_node is not None and prev_node.text.endswith("~}}"):
                text = node.text.lstrip()
            else:
                text = node.text
            
            self.extend_prefix(text)
            return ""
        
        elif node.expr_name == 'comment':
            return node.text
        
        elif node.expr_name == 'slim_comment':
            return node.text

        elif node.expr_name == 'command_args':
            visited_children = [await self.visit(child) for child in node.children]
            return visited_children

        elif node.expr_name == 'command_arg_and_ws':
            # visited_children = [await self.visit(child) for child in node.children]
            return await self.visit(node.children[1]) #visited_children[1]

        elif node.expr_name == 'positional_command_arg':
            # visited_children = [await self.visit(child) for child in node.children]
            return PositionalArgument(await self.visit(node.children[0]))

        elif node.expr_name == 'named_command_arg':
            # visited_children = [await self.visit(child) for child in node.children]
            return NamedArgument(await self.visit(node.children[0]), await self.visit(node.children[2]))

        elif node.expr_name == 'command_name':
            return node.text

        elif node.expr_name == 'escaped_command':
            self.extend_prefix(node.text[1:])
            return

        elif node.expr_name == 'literal':
            try:
                return ast.literal_eval(node.text)
            except Exception as e:
                raise Exception(f"Error parsing literal: {node.text} ({e})")

        elif node.expr_name == 'command':

            # if execution is already stopped before we start the command we just keep the command text
            if not self.executing:
                self.extend_prefix(node.text)
                return
            
            # mark our position in case we need to rewind
            pos = len(self.prefix)

            # find the command name
            command_head = node.children[1].children[0]
            if command_head.expr_name == 'variable_ref':
                if callable(self.get_variable(command_head.children[0].text)):
                    name = command_head.children[0].text
                else:
                    name = "variable_ref"
            elif command_head.expr_name == 'command_call':
                name = command_head.children[0].text
            else:
                raise Exception("Unknown command head type: "+command_head.expr_name)

            # add the start marker
            escaped_node_text = node.text.replace("$", "&#36;").replace("{", "&#123;").replace("}", "&#125;")
            self.extend_prefix("{{!--"+f"GMARKER_START_{name}${escaped_node_text}$"+"--}}")
            
            # visit our children
            self.block_content.append([])
            visited_children = [await self.visit(child, next_node, prev_node, node, parent_node) for child in node.children]
            self.block_content.pop()
            out = "".join("" if c is None else str(c) for c in visited_children)

            self.extend_prefix(out +  "{{!--" + f"GMARKER_END_{name}$$" + "--}}")

            # if execution became stopped during the command, we append the command text
            if not self.executing:
                # self.reset_prefix(pos)
                self.extend_prefix(node.text)
            return

        elif node.expr_name == 'command_arg_group':
            visited_children = [await self.visit(child) for child in node.children]
            return visited_children[1]

        elif node.expr_name == 'command_call' or node.expr_name == 'variable_ref':
            if node.expr_name == 'command_call':
                visited_children = [await self.visit(child) for child in node.children]
                command_name, args = visited_children
            else:
                command_name = node.text
                args = []
            
            return_value = ""
            if self.variable_exists(command_name):
                command_function = self.get_variable(command_name)

                # we convert a variable reference to a function that returns the variable value
                if node.expr_name == "variable_ref" and not callable(command_function):
                    command_value = command_function
                    command_function = lambda: command_value

                def update_return_value(s):
                    nonlocal return_value
                    if return_value == "":
                        return_value = s
                    
                    # convert to strings if we are concatenating
                    else:
                        return_value += "" if s is None else str(s)

                # If we are a top level command we extend the prefix
                if grandparent_node is not None and grandparent_node.expr_name == "command":
                    partial_output = self.extend_prefix
                
                # otherwise we keep track of output locally so we can return it
                else:
                    partial_output = update_return_value 

                

                # create the arguments for the command
                positional_args = []
                named_args = {}
                for arg in args:
                    if isinstance(arg, PositionalArgument):
                        positional_args.append(arg.value)
                    elif isinstance(arg, NamedArgument):
                        named_args[arg.name] = arg.value
                sig = inspect.signature(command_function)
                if "parser_prefix" in sig.parameters:
                    named_args["parser_prefix"] = strip_markers(self.prefix)
                if "parser" in sig.parameters:
                    named_args["parser"] = self
                if "partial_output" in sig.parameters:
                    named_args["partial_output"] = partial_output
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

                # call the command
                try:
                    if inspect.iscoroutinefunction(command_function):
                        await asyncio.sleep(0) # give other coroutines a chance to run
                        command_output = await command_function(*positional_args, **named_args)
                    else:
                        command_output = command_function(*positional_args, **named_args)
                except StopIteration as ret:
                    command_output = ret.value
                    self.caught_stop_iteration = True

                # call partial output if the command didn't itself (and we are still executing)
                if "partial_output" not in sig.parameters:
                    partial_output(command_output)
            else:
                warnings.warn(f"Command/variable '{command_name}' not found")
                return_value = None
            return return_value

        elif node.expr_name == 'block_command_call':
            command_name, args = [await self.visit(child) for child in node.children]
            return command_name, args

        elif node.expr_name == 'command_block_open':
            return await self.visit(node.children[2])
            # visited_children = [await self.visit(child) for child in node.children]
            # return visited_children[2]

        elif node.expr_name == 'command_block':

            # if execution is already stopped before we start the command block we just return unchanged
            if not self.executing:
                self.extend_prefix(node.text)
                return ""

            # create a block content variable
            block_content = [node.children[1]]
            for child in node.children[2].children:
                if child.text == '':
                    continue
                block_content.append(child.children[0])
                block_content.append(child.children[1])
            self.block_content.append(block_content)

            # get the command name and arguments
            command_name, command_args = await self.visit(node.children[0])

            # mark our position in case we need to rewind
            pos = len(self.prefix)

            # add the start marker
            escaped_node_text = node.text.replace("$", "&#36;").replace("{", "&#123;").replace("}", "&#125;")
            start_marker = "{{!--"+f"GMARKER_START_{command_name}${escaped_node_text}$"+"--}}"
            self.extend_prefix(start_marker)

            if self.variable_exists(command_name):
                command_function = self.get_variable(command_name)
                positional_args = []
                named_args = {}
                for arg in command_args:
                    if isinstance(arg, PositionalArgument):
                        positional_args.append(arg.value)
                    elif isinstance(arg, NamedArgument):
                        named_args[arg.name] = arg.value
                sig = inspect.signature(command_function)
                if "parser_prefix" in sig.parameters:
                    named_args["parser_prefix"] = strip_markers(self.prefix)
                if "parser" in sig.parameters:
                    named_args["parser"] = self
                if "block_content" in sig.parameters:
                    named_args["block_content"] = self.block_content[-1]
                if "partial_output" in sig.parameters:
                    named_args["partial_output"] = self.extend_prefix
                if "parser_node" in sig.parameters:
                    named_args["parser_node"] = node
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
                
                if inspect.iscoroutinefunction(command_function):
                    command_output = await command_function(*positional_args, **named_args)
                else:
                    command_output = command_function(*positional_args, **named_args)

                if "partial_output" not in sig.parameters:
                    self.extend_prefix(command_output)

                # if we stopped execution we need to remove the start marker
                if not self.executing:
                    self.prefix = self.prefix[:pos] + self.prefix[pos+len(start_marker):]
                    return
                
            else:
                command_output = ""

            # pop off the block content after the command call
            self.block_content.pop()

            self.extend_prefix("{{!--" + f"GMARKER_END_{command_name}$$" + "--}}")
            return

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
                visited_children.append(await self.visit(child, inner_next_node, inner_prev_node, node, parent_node))
            # visited_children = [self.visit(child) for child in node.children]
            
            if len(visited_children) == 1:
                return visited_children[0]
            else:
                return "".join("" if c is None else c for c in visited_children)

    def get_variable(self, name, default_value=None):
        parts = re.split(r"\.|\[", name)
        for variables in reversed(self.variable_stack):
            curr_pos = variables
            found = True
            for part in parts:
                if part.endswith("]"):
                    var_part = ast.literal_eval(part[:-1])
                else:
                    var_part = part
                try:
                    next_pos = curr_pos[var_part]
                    next_found = True
                except KeyError:
                    next_found = False
                if next_found:
                    curr_pos = next_pos
                else:
                    found = False
                    break
            if found:
                return curr_pos
        return default_value # variable not found

    def variable_exists(self, name):
        out = self.get_variable(name, 849203984939)
        return out != 849203984939

    def set_variable(self, name, value):
        parts = re.split(r"\.|\[", name)
        # parts = name.split(".")
        found = True
        for variables in reversed(self.variable_stack):
            curr_pos = variables
            found = True
            for part in parts:
                if part.endswith("]"):
                    var_part = ast.literal_eval(part[:-1])
                else:
                    var_part = part
                try:
                    next_pos = curr_pos[var_part]
                    next_found = True
                except KeyError:
                    next_found = False
                
                if next_found:
                    if part == parts[-1]:
                        curr_pos[var_part] = value
                        break
                    else:
                        curr_pos = next_pos
                else:
                    if part == parts[-1] and len(parts) > 1: # setting a new property
                        curr_pos[var_part] = value
                    else:
                        found = False
                    break
            if found:
                break
        if not found:
            assert len(parts) == 1, "Can't set a property of a non-existing variable: " + name
            self.variable_stack[0][name] = value

    def extend_prefix(self, text):
        if text == "":
            return
        prefix_out = str(text)
        self.prefix += prefix_out
        self.program._update_display()
    
    def reset_prefix(self, pos):
        self.prefix = self.prefix[:pos]
        self.program._update_display()
        # TODO: undo the echo if needed

class PositionalArgument:
    def __init__(self, value):
        self.value = value

class NamedArgument:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class StopCompletion(Exception):
    pass

