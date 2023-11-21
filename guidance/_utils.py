import os
import requests
import inspect
import json
import re
import asyncio
import queue
import ast
import types
import itertools
import textwrap
import sys

class TextRange:
    def __init__(self, start, end, lm):
        self.start = start
        self.end = end
        self.lm = lm

    def __str__(self) -> str:
        return str(self.lm)[self.start:self.end]

# class InPlace():
#     """Creates a scope where the LM is in-place or not."""
#     def __init__(self, lm):
#         self.lm = lm
#         self._prev_inplace = lm._inplace
    
#     def __enter__(self):
#         new_lm = self.lm._clone()
#         new_lm._inplace = True
#         return new_lm

#     def __exit__(self, type, value, traceback):
#         InPlace._rec_set_inplace(self.lm, self._prev_inplace)

#     @staticmethod
#     def _rec_set_inplace(lm, value):
#         lm._inplace = value
#         for child in lm._children:
#             InPlace._rec_set_inplace(child, value)

# class Silent():
#     """Creates a scope where the LM is silent or not."""
#     def __init__(self, lm, silent):
#         self.lm = lm
#         self._prev_silent = lm._silent
#         self.silent = silent
    
#     def __enter__(self):
#         if self.silent is not None:
#             self.lm._silent = self.silent

#     def __exit__(self, type, value, traceback):
#         if self.silent is not None:
#             Silent._rec_set_silent(self.lm, self._prev_silent)

#     @staticmethod
#     def _rec_set_silent(lm, value):
#         lm._silent = value
#         for child in lm._children:
#             Silent._rec_set_silent(child, value)

# class Hidden():
#     """Creates a scope where the LM state is optionally hidden from following calls.
    
#     Hidden means that the text inside this scope will not be used as context for
#     later calls.
#     """
#     def __init__(self, lm, hidden):
#         self.lm = lm
#         self.hidden = hidden
    
#     def __enter__(self):
#         self.offset = len(self.lm)

#     def __exit__(self, type, value, traceback):
#         if self.hidden:
#             Hidden._rec_make_hidden(self.lm, self.offset)

#     @staticmethod
#     def _rec_make_hidden(lm, offset):
#         lm.reset(offset, clear_variables=False)
#         for child in lm._children:
#             Hidden._rec_make_hidden(child, offset)

class _Rewrite(ast.NodeTransformer):
    def visit_Constant(self, node):
        # print(node)
        if isinstance(node.value, str) and node.lineno < node.end_lineno:
            self.start_counts[node.lineno-1] += 1
            start_line = self.source_lines[node.lineno-1]
            start_string = start_line[node.col_offset:]
            
            # check for literal multiline strings
            if start_string.startswith("f'''") or start_string.startswith("'''") or start_string.startswith('f"""') or start_string.startswith('"""'):
                
                # track our indentation level
                if self.indentation[node.lineno-1] is None:
                    indent = start_line[:len(start_line) - len(start_line.lstrip())]
                    for i in range(node.lineno-1, node.end_lineno):
                        self.indentation[i] = indent
                indent = self.indentation[node.lineno-1]

                # strip indentation when it is consistent
                lines = node.value.split("\n")
                fail = False
                new_lines = []
                for i,line in enumerate(lines):
                    if (i == 0 and (self.start_counts[node.lineno-1] > 1 or not start_line.endswith("\\"))) or line == "":
                        new_lines.append(line)
                    elif line.startswith(indent):
                        new_lines.append(line[len(indent):])
                    # elif (i == 0 and line.endswith("\\")) or line == "":
                    #     new_lines.append(line)
                    else:
                        fail = True
                        break
                if not fail:
                    node.value = "\n".join(new_lines)

        return node
class normalize_notebook_stdout_stderr():
    '''Remaps stdout and stderr back to their normal selves from what ipykernel did to them.
    
    Based on: https://github.com/ipython/ipykernel/issues/795
    '''

    def __enter__(self):
        normal_stdout = sys.__stdout__.fileno()
        self.restore_stdout = None
        if getattr(sys.stdout, "_original_stdstream_copy", normal_stdout) != normal_stdout:
            self.restore_stdout = sys.stdout._original_stdstream_copy
            sys.stdout._original_stdstream_copy = normal_stdout

        normal_stderr = sys.__stderr__.fileno()
        self.restore_stderr = None
        if getattr(sys.stderr, "_original_stdstream_copy", normal_stderr) != normal_stderr:
            self.restore_stderr = sys.stderr._original_stdstream_copy
            sys.stderr._original_stdstream_copy = normal_stderr

    def __exit__(self, exc_type, exc_value, traceback):
        if self.restore_stdout is not None:
            sys.stderr._original_stdstream_copy = self.restore_stdout
        if self.restore_stderr is not None:
            sys.stderr._original_stdstream_copy = self.restore_stderr

def strip_multiline_string_indents(f):

    source = textwrap.dedent(inspect.getsource(f))
    blanks = '\n' * f.__code__.co_firstlineno # padd the source so the lines in the file line up for the debugger
    source = blanks + '\n'.join(source.splitlines()[1:]) # remove the decorator first line.
    
    # define the external closure variables so f.__closure__ will match our recompiled version
    if len(f.__code__.co_freevars) > 0:
        raise Exception("You currently must use @guidance(dedent=False) for closure functions (function nested within other functions that reference the outer functions variables)!")
        lines = source.split("\n")
        lines[0] = "def __outer__closure_wrap():"
        lines[1] = "    " + ",".join(f.__code__.co_freevars) + " = " + ",".join("None" for _ in f.__code__.co_freevars)
        source = "    \n".join(lines) # TODO: this does not quite work because new_code_obj is now the __outer__closure_wrap() function...could be fixed with work...

    old_code_obj = f.__code__
    old_ast = ast.parse(source)
    r = _Rewrite()
    r.source_lines = source.split("\n")
    r.indentation = [None for l in r.source_lines]
    r.start_counts = [0 for l in r.source_lines]
    # r._avoid_backslashes = True
    new_ast = r.visit(old_ast)
    new_code_obj = compile(new_ast, old_code_obj.co_filename, 'exec')

    # find the code block
    for i in range(len(new_code_obj.co_consts)):
        if str(type(new_code_obj.co_consts[i])) == "<class 'code'>":
            break

    # create a new function based on the modified code
    new_f = types.FunctionType(
        new_code_obj.co_consts[i],
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__
    )
    new_f.__kwdefaults__ = f.__kwdefaults__
    return new_f

class CaptureEvents():
    """Creates a scope where all the events are captured in a queue.
    
    Note that this does not stop the events from being captured by higher level scopes.
    """
    def __init__(self, lm):
        self.lm = lm
    
    def __enter__(self):
        self.lm._event_queue = queue.Queue()
        return self.lm._event_queue

    def __exit__(self, type, value, traceback):
        self.lm._event_queue = None

def load(guidance_file):
    ''' Load a guidance program from the given text file.

    If the passed file is a valid local file it will be loaded directly.
    Otherwise, if it starts with "http://" or "https://" it will be loaded
    from the web.
    '''

    if os.path.exists(guidance_file):
        with open(guidance_file, 'r') as f:
            return f.read()
    elif guidance_file.startswith('http://') or guidance_file.startswith('https://'):
        return requests.get(guidance_file).text
    else:
        raise ValueError('Invalid guidance file: %s' % guidance_file)


def chain(programs, **kwargs):
    ''' Chain together multiple programs into a single program.
    
    This merges them into a single program like: {{>program1 hidden=True}}{{>program2 hidden=True}}
    '''

    from ._program import Program

    new_template = "".join(["{{>program%d hidden=True}}" % i for i in range(len(programs))])
    for i, program in enumerate(programs):
        if isinstance(program, Program):
            kwargs["program%d" % i] = program
        else:
            sig = inspect.signature(program)
            args = ""
            for name, _ in sig.parameters.items():
                args += f" {name}={name}"
            fname = find_func_name(program, kwargs)
            kwargs["program%d" % i] = Program("{{set (%s%s)}}" % (fname, args), **{fname: program})
            # kwargs.update({f"func{i}": program})
    return Program(new_template, **kwargs)


def find_func_name(f, used_names):
    if hasattr(f, "__name__"):
        prefix = f.__name__.replace("<", "").replace(">", "")
    else:
        prefix = "function"

    if prefix not in used_names:
        return prefix
    else:
        for i in range(100):
            fname = f"{prefix}{i}"
            if fname not in used_names:
                return fname


def strip_markers(s):
    """This strips out the comment markers used by guidance."""
    if s is None:
        return None
    return re.sub(r"{{!--G.*?--}}", r"", s, flags=re.MULTILINE | re.DOTALL)

class AsyncIter():    
    def __init__(self, items):    
        self.items = items    

    async def __aiter__(self):    
        for item in self.items:    
            yield item

class TrieOld(object):
    __slots__ = ('children', 'value', 'match_version', 'match', 'partial_match')

    def __init__(self, strings=None, values=None):
        self.children = {}
        self.value = []
        self.match_version = -1
        self.match = False
        self.partial_match = False

        if strings is not None:
            for i,s in enumerate(strings):
                self.insert(s, None if values is None else values[i])

    def insert(self, s, value):
        if len(s) == 0:
            self.value.append(value)
        else:
            first_char = s[0]
            if first_char not in self.children:
                self.children[first_char] = Trie()
            self.children[first_char].insert(s[1:], value)

    def values(self, prefix):
        if prefix == "":
            return [self.value] + list(itertools.chain.from_iterable(self.children[k].values(prefix) for k in self.children))
        else:
            return self.children[prefix[0]].values(prefix[1:])

    def __setitem__(self, key, value):
        if len(key) == 0:
            self.value = value
        else:
            if key[0] not in self.children:
                self.children[key[0]] = Trie()
            self.children[key[0]].__setitem__(key[1:], value)

    def __contains__(self, key):
        return self.__getitem__(key) is not None

    def __getitem__(self, key):
        if len(key) == 0:
            return self.value
        elif key[0] in self.children:
            self.children[key[0]].__getitem__(key[1:])
        else:
            return None

class Trie(object):
    __slots__ = ('children', 'value', 'match_version', 'match', 'partial_match', 'parent', 'flag')

    def __init__(self, strings=None, values=None, parent=None):
        self.children = {}
        self.value = None
        self.match_version = -1
        self.match = False
        self.partial_match = False
        self.parent = parent
        self.flag = None # a spot for user code to store state

        if strings is not None:
            for i,s in enumerate(strings):
                self.insert(s, None if values is None else values[i])

    def insert(self, s, value):
        if len(s) == 0:
            self.value = value
        else:
            first_char = s[0]
            if first_char not in self.children:
                self.children[first_char] = Trie(parent=self)
            self.children[first_char].insert(s[1:], value)

    def values(self, prefix):
        if prefix == "":
            sub_values = list(itertools.chain.from_iterable(self.children[k].values(prefix) for k in self.children))
            if self.value is not None:
                sub_values.append(self.value)
            return sub_values
        else:
            return self.children[prefix[0]].values(prefix[1:])

    def __setitem__(self, key, value):
        if len(key) == 0:
            self.value = value
        else:
            if key[0] not in self.children:
                self.children[key[0]] = Trie(parent=self)
            self.children[key[0]].__setitem__(key[1:], value)

    def __contains__(self, key):
        return self.__getitem__(key) is not None

    def __getitem__(self, key):
        if len(key) == 0:
            return self
        elif key[0] in self.children:
            return self.children[key[0]].__getitem__(key[1:])
        else:
            return None
        
class ByteTrie(object):
    __slots__ = ('children', 'value', 'match_version', 'match', 'partial_match', 'parent', 'flag', 'log_prob')

    def __init__(self, byte_strings=None, values=None, parent=None):
        self.children = {}
        self.value = None
        self.match_version = -1
        self.match = False
        self.partial_match = False
        self.parent = parent
        self.flag = None # a spot for user code to store state
        self.log_prob = 0

        if byte_strings is not None:
            for i,s in enumerate(byte_strings):
                self.insert(s, None if values is None else values[i])

    def insert(self, s, value):
        if len(s) == 0:
            self.value = value
        else:
            first_byte = s[0:1]
            if first_byte not in self.children:
                self.children[first_byte] = ByteTrie(parent=self)
            self.children[first_byte].insert(s[1:], value)

    # def values(self, prefix):
    #     if prefix == "":
    #         sub_values = list(itertools.chain.from_iterable(self.children[k].values(prefix) for k in self.children))
    #         if self.value is not None:
    #             sub_values.append(self.value)
    #         return sub_values
    #     else:
    #         return self.children[prefix[0]].values(prefix[1:])

    # def __setitem__(self, key, value):
    #     if len(key) == 0:
    #         self.value = value
    #     else:
    #         if key[0] not in self.children:
    #             self.children[key[0]] = Trie(parent=self)
    #         self.children[key[0]].__setitem__(key[1:], value)

    # def __contains__(self, key):
    #     return self.__getitem__(key) is not None

    # def __getitem__(self, key):
    #     if len(key) == 0:
    #         return self
    #     elif key[0] in self.children:
    #         return self.children[key[0]].__getitem__(key[1:])
    #     else:
    #         return None

class ContentCapture:
    def __init__(self, variable_stack, hidden=False):
        self._hidden = hidden
        self._variable_stack = variable_stack
    
    def __enter__(self):
        self._pos = len(self._variable_stack["@raw_prefix"])
        if self._hidden:
            self._variable_stack.push({"@raw_prefix": self._variable_stack["@raw_prefix"]})
        return self

    def __exit__(self, type, value, traceback):
        if self._hidden:
            new_content = str(self)
            self._variable_stack.pop()
            self._variable_stack["@raw_prefix"] += "{{!--GHIDDEN:"+new_content.replace("--}}", "--_END_END")+"--}}"

    def __str__(self):
        return strip_markers(self._variable_stack["@raw_prefix"][self._pos:])
    
    def __iadd__(self, other):
        if other is not None:
            self._variable_stack["@raw_prefix"] += other
        return self
    
    def inplace_replace(self, old, new):
        """Replace all instances of old with new in the captured content."""
        self._variable_stack["@raw_prefix"] = self._variable_stack["@raw_prefix"][:self._pos] + self._variable_stack["@raw_prefix"][self._pos:].replace(old, new)

class JupyterComm():
    def __init__(self, target_id, ipython_handle, callback=None, on_open=None, mode="register"):
        from ipykernel.comm import Comm

        self.target_name = "guidance_interface_target_" + target_id
        # print("TARGET NAME", self.target_name)
        self.callback = callback
        self.jcomm = None
        self.ipython_handle = ipython_handle
        self.addd = 1
        self.send_queue = asyncio.Queue()
        self.open_event = asyncio.Event()
        self.is_open = False
        asyncio.get_event_loop().create_task(self._send_loop())
        if mode == "register":
            # log("REGISTERING", self.target_name)
            # asyncio.get_event_loop().create_task(self._register())
            def comm_opened(comm, open_msg):
                # log("OPENED")
                self.addd = 2
                self.jcomm = comm
                self.is_open = True
                self.jcomm.on_msg(self._fire_callback)
                self.open_event.set()
                self._fire_callback({"content": {"data": {"event": "opened"}}})

            self.ipython_handle.kernel.comm_manager.register_target(self.target_name, comm_opened)
            # get_ipython().kernel.comm_manager.register_target(self.target_name, comm_opened) # noqa: F821
        elif mode == "open":
            # log("OPENING", self.target_name)
            self.jcomm = Comm(target_name=self.target_name)
            self.jcomm.on_msg(self._fire_callback)
            # self._fire_callback({"content": {"data": "opened"}})
        else:
            raise Exception("Passed mode must be either 'open' or 'register'!")

    # async def _register(self):
    #     def comm_opened(comm, open_msg):
    #         #log("OPENED")
    #         self.addd = 2
    #         self.jcomm = comm
    #         self.jcomm.on_msg(self._fire_callback)
    #         self.open_event.set()
    #         self._fire_callback({"content": {"data": {"event": "opened"}}})
    #     get_ipython().kernel.comm_manager.register_target(self.target_name, comm_opened)

    # def send(self, data, wait=False):
    #     self.send_queue.append(data)
    #     if self.jcomm is None:
    #         return
    #     for d in self.send_queue:
    #         self.jcomm.send({"data": json.dumps(d)})
    #     self.send_queue = []

    def clear_send_queue(self):
        while not self.send_queue.empty():
            self.send_queue.get_nowait()
            self.send_queue.task_done()

    def _fire_callback(self, msg):
        self.callback(msg["content"]["data"])

    def send(self, data):
        self.send_queue.put_nowait(data)

    async def _send_loop(self):
        while True:
            # log("SENDING_LOOP")
            if self.jcomm is None:
                self.open_event.clear()
                await self.open_event.wait()
            data = await self.send_queue.get()
            # log("SENDING_LOOP got one!")
            self.jcomm.send({"data": json.dumps(data)})

    # async def _waiting_send(self, data):
    #     #log("SENDING", self.jcomm, data)

    #     # await the open event if needed
    #     if self.jcomm is None:
    #         self.open_event.clear()
    #         await self.open_event.wait()
    #     #log("SENDING_now", self.jcomm, data)
    #     self.jcomm.send({"data": json.dumps(data)}) # we encode the JSON so iPython doesn't mess it up


# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def escape_template_block(text):
    return text.replace("$", "&#36;").replace("{", "&#123;").replace("}", "&#125;")


def unescape_template_block(text):
    return text.replace("&#36;", "$").replace("&#123;", "{").replace("&#125;", "}")
