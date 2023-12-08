import os
import requests
import inspect
import json
import asyncio
import queue
import ast
import types
import textwrap
import sys
import numpy as np

class _Rewrite(ast.NodeTransformer):
    def visit_Constant(self, node):
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

class AsyncIter():    
    def __init__(self, items):    
        self.items = items    

    async def __aiter__(self):    
        for item in self.items:    
            yield item

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


def log_softmax(array: np.ndarray, axis: int = -1) -> np.ndarray:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_softmax.html
    array_maxs: np.ndarray = np.amax(array, axis=axis, keepdims=True)
    if array_maxs.ndim > 0:
        array_maxs[~np.isfinite(array_maxs)] = 0
    elif not np.isfinite(array_maxs):
        array_maxs = 0
    subtract_maxs = array - array_maxs
    exp = np.exp(subtract_maxs)
    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        summed = np.sum(exp, axis=axis, keepdims=True)
        out = np.log(summed)
    return subtract_maxs - out


def softmax(array: np.ndarray, axis: int = -1) -> np.ndarray:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
    array_maxs = np.amax(array, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(array - array_maxs)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)
