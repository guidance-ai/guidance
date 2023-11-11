import inspect
import re
import html
import uuid
# import parsimonious
import logging
import copy
import asyncio
import pathlib
import os
import time
import datetime
import nest_asyncio
# from .llms import _openai
from . import _utils
from ._program_executor import ProgramExecutor
from . import commands
import guidance
log = logging.getLogger(__name__)

# load the javascript client code
file_path = pathlib.Path(__file__).parent.parent.absolute()
with open(file_path / "guidance" / "resources" / "main.js", encoding="utf-8") as f:
    js_data = f.read()

class Log:
    def __init__(self) -> None:
        self._entries = []
    
    def append(self, entry):
        if not hasattr(entry, "time"):
            entry["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._entries.append(entry)

    def __str__(self) -> str:
        string_entries = []
        for entry in self._entries:
            type_str = "["+entry.get("type", "")+"]"
            string_entries.append(
                f"{entry.get('time', '')}{type_str: >10s}{entry.get('name', ''): >15s}: " + str({k:v for k,v in entry.items() if k not in ["type", "name", "time"]})
            )
        return "\n".join(string_entries)
    
    def __repr__(self) -> str:
        return str(self)
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __getitem__(self, key):
        return self._entries[key]
    
    def copy(self):
        new_log = Log()
        new_log._entries = [copy.copy(v) for v in self._entries]
        return new_log

class Program:
    ''' A program template that can be compiled and executed to generate a new filled in (executed) program.

    Note that as the template gets executed {{!-- handlebars comment markers --}} get left in
    the generated output to mark where template tags used to be.
    '''

    def __init__(self, text, llm=None, cache_seed=0, logprobs=None, silent=None, async_mode=False, stream=None, caching=None, await_missing=False, log=None, **kwargs):
        """ Create a new Program object from a program string.

        Parameters
        ----------
        text : str
            The program string to use as a guidance template.
        llm : guidance.llms.LLM (defaults to guidance.llm)
            The language model to use for executing the program.
        cache_seed : int (default 0) or None
            The seed to use for the cache. If you want to use the same cache for multiple programs
            you can set this to the same value for all of them. Set this to None to disable caching.
            Caching is enabled by default, and saves calls that have tempurature=0, and also saves
            higher temperature calls but uses different seed for each call.
        logprobs : int or None (default)
            The number of logprobs to return from the language model for each token. (not well supported yet,
            since some endpoints don't support it)
        silent : bool (default None)
            If True, the program will not display any output. This is useful for programs that are
            only used to generate variables for other programs. If None we automatically set this based
            on if we are streaming and if we are in interactive mode.
        async_mode : bool (default False)
            If True, the program will be executed asynchronously. This is useful for programs that
            take a long time to run, or that need to be run in parallel.
        stream : bool (default None)
            If True, the program will try to stream all the results from the LLM token by token. If None
            streaming will be enabled if is needed for funtionality. (Warning: this param may change a bit in the future)
        caching : bool (default None)
            If True, the program will cache the results of the LLM. If False, it will not cache the results.
            If None, it will use the default caching setting from the LLM.
        await_missing : bool (default False)
            If True, the program will automatically await any missing variables. This means the program
            will stop executation at that point and return a paritally executed program. This is useful
            for executing programs on different machines, for example shipping a program to a GPU machine
            then waiting for the results to come back for any local processing, then shipping it back to
            the GPU machine to continue execution.
        log : bool or Log
            If True, the program will log all the commands that are executed into the `program.log` property.
            If a Log object is passed in, it will be used as the log instead of creating a new one.
        """

        # see if we were given a raw function instead of a string template
        # if so, convert it to a string template that calls the function
        if not isinstance(text, str) and callable(text):
            sig = inspect.signature(text)
            args = ""
            for name,_ in sig.parameters.items():
                args += f" {name}={name}"
            fname = _utils.find_func_name(text, kwargs)
            kwargs[fname] = text
            text = "{{set (%s%s)}}" % (fname, args)
        
        # save the given parameters
        self._text = text
        self.llm = llm or getattr(guidance, "llm", None)
        self.cache_seed = cache_seed
        self.caching = caching
        self.logprobs = logprobs
        self.async_mode = async_mode
        self.silent = silent
        self.stream = stream
        self.await_missing = await_missing
        self.log = log
        if self.silent is None:
            self.silent = self.stream is True or not _utils.is_interactive()

        # build or capture the log
        if self.log is True:
            self.log = Log()
        
        # set our variables
        self._variables = {}
        self._variables.update(_built_ins)
        self._variables.update({
            "llm": llm
        })
        kwargs.pop("self", None)
        kwargs = dict(kwargs)
        for k in list(kwargs.keys()): # handle @varname syntax
            if k.startswith("_AT_"):
                kwargs["@"+k[4:]] = kwargs[k]
                kwargs.pop(k)
        self._variables.update(kwargs)
        
        # set internal state variables
        self._id = str(uuid.uuid4())
        
        self._comm = None # front end communication object
        self._executor = None # the ProgramExecutor object that is running the program
        self._last_display_update = 0 # the last time we updated the display (used for throttling updates)
        self._execute_complete = asyncio.Event() # fires when the program is done executing to resolve __await__
        self._emit_stream_event = asyncio.Event() # fires when we need to emit a stream event
        self._exception = None # if the program finished with an exception its stored here
        self._displaying = not self.silent # if we are displaying we need to update the display as we execute
        self._displayed = False # marks if we have been displayed in the client yet
        self._displaying_html = False # if we are displaying html (vs. text)
        self._tasks = [] # list of children tasks
    
    def __repr__(self):
        return self.text
    
    def __getitem__(self, key):
        return self._variables[key]
    
    def _interface_event(self, msg):
        """ Handle an event from the front end.
        """
        if msg["event"] == "stop":
            self._executor.stop()
        elif msg["event"] == "opened":
            pass # we don't need to do anything here because the first time we display we'll send the html
        pass

    async def _await_finish_execute(self):
        """Used by self.__await__ to wait for the program to complete."""
        try:
            await self._execute_complete.wait() # wait for the program to finish executing

        except asyncio.CancelledError:
            # if this task gets canceled, cancel all sub-tasks
            for task in self._tasks:
                task.cancel()

        # if the program finished executing with an exception
        # re-raise the exception in the main coroutine
        if self._exception:
            raise self._exception

        return self

    def __await__(self):
        return self._await_finish_execute().__await__()
    
    def __aiter__(self):
        """Return an async iterator that yields the program in partial states as it is run."""
        return self._stream_run_async()
        
    def __call__(self, **kwargs):
        """Execute this program with the given variable values and return a new executed/executing program.

        Note that the returned program might not be fully executed if `stream=True`. When streaming you need to
        use the python `await` keyword if you want to ensure the program is finished (note that is different than
        the `await` guidance langauge command, which will cause the program to stop execution at that point).
        """

        # merge the given kwargs with the current variables
        kwargs = {**{
            "async_mode": self.async_mode,
            "stream": self.stream,
            "silent": self.silent,
            "cache_seed": self.cache_seed,
            "caching": self.caching,
            "logprobs": self.logprobs,
            "await_missing": self.await_missing,
            "log": self.log.copy() if hasattr(self.log, "copy") else self.log,
            "llm": self.llm,
        }, **kwargs}

        log.debug(f"in __call__ with kwargs: {kwargs}")

        # create a new program object that we will execute in-place
        new_program = Program(
            text=self.marked_text,

            # copy the (non-function) variables so that we don't modify the original program during execution
            # TODO: what about functions? should we copy them too?
            **{**{k: v if callable(v) else copy.deepcopy(v) for k,v in self._variables.items()}, **kwargs}
        )

        # create an executor for the new program (this also marks the program as executing)
        new_program._executor = ProgramExecutor(new_program)
        
        # if we are in async mode, schedule the program in the current event loop
        if new_program.async_mode:
            loop = asyncio.get_event_loop()
            assert loop.is_running(), "The program is in async mode but there is no asyncio event loop running! Start one and try again."
            execute_task = loop.create_task(new_program.execute())
            new_program._tasks.append(execute_task)

        # if we are not in async mode, we need to create a new event loop and run the program in it until it is done
        else:

            # apply nested event loop patch if needed
            try:
                other_loop = asyncio.get_event_loop()
                nest_asyncio.apply(other_loop)
            except RuntimeError:
                pass
            
            loop = asyncio.new_event_loop()
            if new_program.stream:
                return self._stream_run(loop, new_program)
            else:
                loop.run_until_complete(new_program.execute())

        return new_program
    
    def get(self, key, default=None):
        """Get the value of a variable by name."""
        return self._variables.get(key, default)
    
    def _stream_run(self, loop, new_program):
        """This feels a bit hacky at the moment. TODO: clean this up."""

        # add the program execution to the event loop
        execute_task = loop.create_task(new_program.execute())
        new_program._tasks.append(execute_task)

        # run the event loop until the program is done executing
        while new_program._executor is not None:
            try:
                loop.run_until_complete(execute_task) # this will stop each time the program wants to emit a new state
            except RuntimeError as e:
                # we don't mind that the task is not yet done, we will restart the loop
                if str(e) != "Event loop stopped before Future completed.":
                    raise e
            if getattr(loop, "_stopping", False):
                loop._stopping = False # clean up the stopping flag
            if new_program._executor is not None and new_program._executor.executing:
                try:
                    yield new_program
                except GeneratorExit:
                    # this will cause the program to stop executing and finish as a valid partial execution
                    if new_program._executor.executing:
                        new_program._executor.executing = False
        yield new_program

        # cancel all tasks and close the loop
        for task in self._tasks:
            task.cancel()
        loop.run_until_complete(asyncio.sleep(0)) # give the loop a chance to cancel the tasks

        # TODO: do we really want to close the loop? what if it is used by others?
        loop.close() # we are done with the loop (note that the loop is already stopped)

    async def _stream_run_async(self):

        # run the event loop until the program is done executing
        while self._executor is not None:
            if self._executor.executing:
                await self._emit_stream_event.wait()
                self._emit_stream_event.clear()
            try:
                yield self
            except GeneratorExit as e:
                # this will cause the program to stop executing and finish as a valid partial execution
                if self._executor.executing:
                    self._executor.executing = False
                await self._execute_complete.wait()

                raise e
        yield self


    async def execute(self):
        """ Execute the current program.

        Note that as execution progresses the program will be incrementally converted
        from a template into a completed string (with variables stored). At each point
        in this process the current template remains valid.
        """

        log.debug(f"Executing program (self.async_mode={self.async_mode}, self.silent={self.silent}, self._displaying_html={self._displaying_html})")
        
        # if we are already displaying html, we need to yield to the event loop so the jupyter comm can initialize
        if self._displaying_html:
            await asyncio.sleep(0)
        
        # run the program and capture the output
        try:
            if self.llm is None:
                await self._executor.run(None)
            else:
                with self.llm.session(asynchronous=True) as llm_session:
                    await self._executor.run(llm_session)
            self._text = self._variables["@raw_prefix"]

        # if the execution failed, capture the exception so it can be re-raised
        # in the main coroutine
        except Exception as exception:
            self._exception = exception

        finally:
            # delete the executor and so mark the program as not executing
            self._executor = None

            # fire an event noting that execution is complete (this will release any await calls waiting on the program)
            self._execute_complete.set()

    def __getitem__(self, key):
        return self._variables[key]
    
    def __contains__(self, key):
        return key in self._variables
    
    def __delitem__(self, key):
        del self._variables[key]
    
    def variables(self, built_ins=False, show_hidden=False):
        """ Returns a dictionary of the variables in the program.

        Parameters
        ----------
        built_ins : bool
            If True, built-in variables will be included in the returned dictionary.
        show_hidden : bool
            If True, hidden variables will be included in the returned dictionary.
        """
        out = {}
        for k,v in self._variables.items():
            if show_hidden or not k.startswith("_"):
                if built_ins or not (k in _built_ins and callable(_built_ins[k])):
                    out[k] = v
            
        return out
    
    @property
    def text(self):
        # strip out the markers for the unformatted output
        return _utils.strip_markers(self.marked_text)
    
    @property
    def marked_text(self):
        if self._executor is not None:
            return self._variables["@raw_prefix"]
        else:
            return self._text
    

def add_spaces(s):
    """ This adds spaces so the browser will show leading and trailing newlines.
    """
    if s.startswith("\n"):
        s = " " + s
    if s.endswith("\n"):
        s = s + " "
    return s

_built_ins = {
    "gen": commands.gen,
    "each": commands.each,
    "geneach": commands.geneach,
    "select": commands.select,
    "if": commands.if_,
    "unless": commands.unless,
    "add": commands.add,
    "BINARY_OPERATOR_+": commands.add,
    "subtract": commands.subtract,
    "BINARY_OPERATOR_-": commands.subtract,
    "multiply": commands.multiply,
    "BINARY_OPERATOR_*": commands.multiply,
    "strip": commands.strip,
    "block": commands.block,
    "set": commands.set,
    "await": commands.await_,
    "role": commands.role,
    "user": commands.user,
    "system": commands.system,
    "assistant": commands.assistant,
    "function": commands.function,
    "break": commands.break_,
    "equal": commands.equal,
    "BINARY_OPERATOR_==": commands.equal,
    "notequal": commands.notequal,
    "BINARY_OPERATOR_!=": commands.notequal,
    "greater": commands.greater,
    "BINARY_OPERATOR_>": commands.greater,
    "less": commands.less,
    "BINARY_OPERATOR_<": commands.less,
    "contains": commands.contains,
    "parse": commands.parse,
    "callable": commands.callable,
    "len": commands.len,
    "range": commands.range,
    "UNARY_OPERATOR_not": commands.not_,
}
