import os
import requests
import inspect
import time
import json
import re
import datetime
import asyncio

log_file = None


def open_log_file():
    global log_file
    if log_file is None:
        log_file = open("log.txt", "a")
    return log_file


def close_log_file():
    global log_file
    if log_file is not None:
        log_file.close()
        log_file = None


def log(*args):
    # print(*args)
    log_file = open_log_file()
    print(datetime.datetime.now().strftime("%H:%M:%S"), *args, file=log_file)
    log_file.flush()
    # Currently never gets closed


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
    return re.sub(r"{{!--G.*?--}}", r"", s, flags=re.MULTILINE | re.DOTALL)


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
