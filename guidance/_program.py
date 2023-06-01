import ast
import inspect
import re
import html
import uuid
import sys
import parsimonious
import logging
import copy
import asyncio
import pathlib
import os
import traceback
import time
import datetime
import nest_asyncio
from .llms import _openai
from . import _utils
from ._program_executor import ProgramExecutor
from . import library
import guidance
log = logging.getLogger(__name__)

# load the javascript client code
file_path = pathlib.Path(__file__).parent.parent.absolute()
with open(file_path / "guidance" / "resources" / "main.js", encoding="utf-8") as f:
    js_data = f.read()

class Program:
    ''' A program template that can be compiled and executed to generate a new filled in (executed) program.

    Note that as the template gets executed {{!-- handlebars comment markers --}} get left in
    the generated output to mark where template tags used to be.
    '''

    def __init__(self, text, llm=None, cache_seed=0, logprobs=None, silent=None, async_mode=False, stream=None, caching=None, await_missing=False, **kwargs):
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
        if self.silent is None:
            self.silent = self.stream is True or not _utils.is_interactive()
        
        # set our variables
        self._variables = {}
        self._variables.update(_built_ins)
        self._variables.update({
            "llm": llm
        })
        self._variables.update(kwargs)
        
        # set internal state variables
        self._id = str(uuid.uuid4())
        
        self._comm = None # front end communication object
        self._executor = None # the ProgramExecutor object that is running the program
        self._last_display_update = 0 # the last time we updated the display (used for throttling updates)
        self._execute_complete = asyncio.Event() # fires when the program is done executing to resolve __await__
        self._emit_stream_event = asyncio.Event() # fires when we need to emit a stream event
        self._displaying = not self.silent # if we are displaying we need to update the display as we execute
        self._displayed = False # marks if we have been displayed in the client yet
        self._displaying_html = False # if we are displaying html (vs. text)

        # throttle the display updates
        if os.environ.get("VSCODE_CWD", None) is not None:
            self.display_throttle_limit = 0.1 # VSCode has a bug that causes flashing, so we slow down the display
        else:
            self.display_throttle_limit = 0.1 # the minimum time between display updates
        self.update_display = DisplayThrottler(self._update_display, self.display_throttle_limit)

        # see if we are in an ipython environment
        try:
            from IPython import get_ipython
            self._ipython = get_ipython()
        except:
            self._ipython = None
        
        # if we are echoing in ipython we assume we can display html
        if self._ipython and not self.silent:
            self._displaying_html = True
    
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

    def _ipython_display_(self):
        """ Display the program in the ipython notebook.
        """

        log.debug(f"displaying program in _ipython_display_ with self._comm={self._comm}, self.id={self._id}")
        
        # mark that we are displaying (and so future execution updates should be displayed)
        self._displaying = True
        self._displaying_html = True
        
        # build and display the html
        html = self._build_html(self.marked_text)
        self._display_html(html)
        

    async def _await_finish_execute(self):
        """Used by self.__await__ to wait for the program to complete."""
        await self._execute_complete.wait() # wait for the program to finish executing
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
            loop.create_task(new_program.update_display.run()) # start the display updater
            loop.create_task(new_program.execute())

        # if we are not in async mode, we need to create a new event loop and run the program in it until it is done
        else:

            # apply nested event loop patch if needed
            try:
                other_loop = asyncio.get_event_loop()
                nest_asyncio.apply(other_loop)
            except RuntimeError:
                pass
            
            loop = asyncio.new_event_loop()
            loop.create_task(new_program.update_display.run()) # start the display updater
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
        for task in asyncio.all_tasks(loop=loop):
            task.cancel()
        loop.run_until_complete(asyncio.sleep(0)) # give the loop a chance to cancel the tasks
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

    def _update_display(self, last=False):
        """Updates the display with the current marked text after debouncing.

        Parameters
        ----------
        last : bool
            If True, this is the last update and we should clear the send queue and prepare the
            UI for saving etc.
        force : bool
            If True, we will update the display even if it would otherwise be throttled.
        """

        log.debug(f"Updating display (last={last}, self._displaying={self._displaying}, self._comm={self._comm})")

        
        if self.stream:
            if self.async_mode:
                # if we are streaming in async mode then we set the event to let the generator know it can yield
                self._emit_stream_event.set()
                
            else:
                # if we are streaming not in async mode then we pause the event loop to let the generator
                # that is controlling execution return (it will restart the event loop when it is ready)
                if self._executor is not None:
                    asyncio.get_event_loop().stop()

        # this is always called during execution, and we only want to update the display if we are displaying
        if not self._displaying:
            return
        
        # debounce the display updates
        # now = time.time()
        # log.debug(now - self._last_display_update)
        # debounce_delay = self.display_throttle_limit if self._comm and self._comm.is_open else self.display_throttle_limit_low
        # if last or (now - self._last_display_update > debounce_delay):
        if self._displaying_html:
            out = self._build_html(self.marked_text)
            
            # clear the send queue if this is the last update
            if last and self._comm:
                self._comm.clear_send_queue()
            
            # send an update to the front end client if we have one...
            # TODO: we would like to call `display` for the last update so NB saving works, but see https://github.com/microsoft/vscode-jupyter/issues/13243 
            if self._displayed and self._comm and self._comm.is_open: #(not last or self._comm.is_open):
                log.debug(f"Updating display send message to front end")
                # log.debug(out)
                self._comm.send({"replace": out})
                if last:
                    self._comm.send({"event": "complete"})
            
            # ...otherwise dump the client to the front end
            else:
                log.debug(f"Updating display dump to front end")
                from IPython.display import clear_output, display
                if self._displayed:
                    clear_output(wait=True) # TODO: should use wait=True but that doesn't work in VSCode until after the April 2023 release

                self._display_html(out)
        
        self._last_display_update = time.time()

    def _display_html(self, html):
        from IPython.display import display

        # create the comm object if we don't have one
        if self._comm is None:
            self._comm = _utils.JupyterComm(self._id, self._ipython, self._interface_event)
        
        # dump the html to the front end
        html = f"""<div id="guidance-stop-button-{self._id}" style="cursor: pointer; margin: 0px; display: none; float: right; padding: 3px; border-radius: 4px 4px 4px 4px; border: 0px solid rgba(127, 127, 127, 1); padding-left: 10px; padding-right: 10px; font-size: 13px; background-color: rgba(127, 127, 127, 0.25);">Stop program</div><div id="guidance-content-{self._id}">{html}</div>
<script type="text/javascript">{js_data}; window._guidanceDisplay("{self._id}");</script>"""
        display({"text/html": html}, display_id=self._id, raw=True, clear=True, include=["text/html"])
        self._displayed = True

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
        if self.llm is None:
            await self._executor.run(None)
        else:
            with self.llm.session(asynchronous=True) as llm_session:
                await self._executor.run(llm_session)
        self._text = self._executor.prefix

        # delete the executor and so mark the program as not executing
        self._executor = None

        # update the display with the final output
        self.update_display(last=True)
        await self.update_display.done()

        # fire an event noting that execution is complete (this will release any await calls waiting on the program)
        self._execute_complete.set()
    
    def __getitem__(self, key):
        return self._variables[key]
    
    def __contains__(self, key):
        return key in self._variables
    
    def __delitem__(self, key):
        del self._variables[key]
    
    def variables(self, built_ins=False):
        """ Returns a dictionary of the variables in the program.

        Parameters
        ----------
        built_ins : bool
            If True, built-in variables will be included in the returned dictionary.
        """
        return {k: v for k,v in self._variables.items() if built_ins or not (k in _built_ins and callable(_built_ins[k]))}
    
    @property
    def text(self):
        # strip out the markers for the unformatted output
        return _utils.strip_markers(self.marked_text)
    
    @property
    def marked_text(self):
        if self._executor is not None:
            return self._executor.prefix
        else:
            return self._text
    
    def _build_html(self, text, last=False):
        output = text

        def undo_html_encode(x):
            return x.replace("&amp;#123;", "{").replace("&amp;#125;", "}").replace("&amp;#36;", "$")

        def start_generate_or_select(x):
            no_echo = "echo=False" in x.group(1)
            alpha = 1.0 if no_echo else 1.0
            
            # script that toggles the viisibility of the next element
            click_script = 'var e = this.nextElementSibling; if (e.style.display == "inline") { e.style.display = "none"; this.style.borderRight = "1px solid rgba(0, 165, 0, 0.25)"; } else { e.style.display = "inline"; this.style.borderRight = "0px";}'

            if no_echo:
                out = f'''<div style='background-color: rgba(0, 165, 0, 0.25); border-radius: 4px 0px 0px 4px; border: 1px solid rgba(0, 165, 0, 1); padding-left: 3px; padding-right: 3px; user-select: none; color: rgb(0, 165, 0, 1.0); display: inline; font-weight: normal; cursor: pointer' onClick='{click_script}'>no echo</div>'''
                out += "<span style='background-color: rgba(0, 165, 0, 0.25); opacity: {}; display: none;' title='{}'>".format(alpha, undo_html_encode(x.group(1)))
            else:
                out = "<span style='background-color: rgba(0, 165, 0, 0.25); opacity: {}; display: inline;' title='{}'>".format(alpha, undo_html_encode(x.group(1)))
            return out
        
        def start_each(x):
            no_echo = "echo=False" in x.group(1)
            alpha = 0.5 if no_echo else 1.0
            color = "rgba(165, 165, 165, 0.1)" #if "geneach" not in x.group(1) else "rgba(0, 165, 0, 0.1)"
            return "<span style='opacity: {}; display: inline; background-color: {};' title='{}'>".format(alpha, color, undo_html_encode(x.group(1)))
        
        def start_block(x):
            escaped_tag = undo_html_encode(x.group(1))
            if "hidden=True" in escaped_tag:
                display = "inline" # none (we actively stip hidden tags right now so we don't need this until we support the UX to show hidden stuff)
            else:
                display = "inline"
            return f"<span style='background-color: rgba(165, 165, 165, 0.1); display: {display};' title='{escaped_tag}'>"
        
        def role_box(x):
            # name = x.group(3).lower() # standardize to lowercase for display
            # content = x.group(4)

            content = x.group(3)
            tag_text = undo_html_encode(x.group(2))
            role_name = x.group(1)
            
            # if we have a generic role tag then the role name is an attribute
            if role_name == "role":
                role_name = re.search(r"name=([^ ]*)", tag_text).group(1)
            
            start_pattern = html.escape(self.llm.role_start(role_name)).replace("|", r"\|")
            end_pattern = html.escape(self.llm.role_end(role_name)).replace("|", r"\|")

            # strip the start and end patterns from the content
            content = re.sub("^" + start_pattern, "", content, flags=re.DOTALL)
            content = re.sub(end_pattern + "$", "", content, flags=re.DOTALL)

            
            # one div that contains two divs, where the left of the two inner divs has a fixed width of 100px
            # """<div style='display: flex;'>
            #     <div style='width: 100px; border-right: 1px solid rgba(127, 127, 127, 0.2); padding-right: 5px; margin-right: 5px;'>{name}</div>
            #     <div>{content}</div>
            # </div>"""

            # return f'''<div style="border-left: 1px solid rgba(127, 127, 127, 0.2); margin-top: 10px; padding-left: 5px;"><span style="color: rgba(127,127,127,0.5)">{name}</span>
# {content}</div>'''

            return f"<div style='display: flex; border-bottom: 1px solid rgba(127, 127, 127, 0.2); align-items: center;'><div style='flex: 0 0 80px; opacity: 0.5;'>{role_name.lower()}</div><div style='flex-grow: 1; padding: 5px; padding-top: 10px; padding-bottom: 10px; margin-top: 0px; white-space: pre-wrap; margin-bottom: 0px;'>{content}</div></div>"

        display_out = html.escape(output)
        # log.debug(display_out)

        
        # start_pattern = html.escape(self.llm.role_start("(.*?)")).replace("|", r"\|")
        # end_pattern = html.escape(self.llm.role_end("(.*?)")).replace("|", r"\|")
        # display_out = re.sub(r"[\s]+({{!--.*?--}})?"+start_pattern, r"\1"+start_pattern.replace("(.*?)", r"\1").replace(r"\|", "|"), display_out, flags=re.DOTALL)
        # display_out = re.sub(start_pattern + "(.*?)" + end_pattern, role_box, display_out, flags=re.DOTALL)
        # log.debug(display_out)

        # strip out hidden blocks (might want to make a better UI for this at some point)
        display_out = re.sub(r"{{!--GMARKER_START[^}]*--}}{{!--GHIDDEN:(.*?)--}}{{!--GMARKER_END[^}]*--}}", "", display_out, flags=re.DOTALL)
        
        # if we have role markers, we wrap them in special formatting
        if re.search(r"{{!--GMARKER_START_(role|system|user|assistant)", display_out) is not None:

            # start_pattern = html.escape(self.llm.role_start("assistant")).replace("|", r"\|").replace(r"assistant", r"([^\n]*)").replace(r"ASSISTANT", r"([^\n]*)")
            # end_pattern = html.escape(self.llm.role_end("assistant")).replace("|", r"\|").replace(r"assistant", r"([^\n]*)").replace(r"ASSISTANT", r"([^\n]*)")
            
            # strip whitespace before role markers
            display_out = re.sub(r"\s*{{!--GMARKER_START_(role|system|user|assistant)\$(.*?)--}}", r"{{!--GMARKER_START_\1$\2--}}", display_out, flags=re.DOTALL)

            # strip whitespace after role markers
            # TODO: support end_patterns with capture groups
            display_out = re.sub(r"{{!--GMARKER_END_(role|system|user|assistant)\$(.*?)--}}\s*", r"{{!--GMARKER_END_\1$\2--}}", display_out, flags=re.DOTALL)

            # wrap role markers in nice formatting
            display_out = re.sub(r"{{!--GMARKER_START_(role|system|user|assistant)\$(.*?)--}}" + "(.*?)" + r"{{!--GMARKER_END_(role|system|user|assistant)\$(.*?)--}}", role_box, display_out, flags=re.DOTALL)

            # wrap unfinished role markers in nice formatting
            display_out = re.sub(r"{{!--GMARKER_START_(role|system|user|assistant)\$(.*?)--}}" + "(.*)", role_box, display_out, flags=re.DOTALL)
        
        display_out = re.sub(r"(\{\{generate.*?\}\})", r"<span style='background-color: rgba(0, 165, 0, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
        display_out = re.sub(r"(\{\{#select\{\{/select.*?\}\})", r"<span style='background-color: rgba(0, 165, 0, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
        display_out = re.sub(r"(\{\{#each [^'\"].*?\{\{/each.*?\}\})", r"<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
        display_out = re.sub(r"(\{\{(?!\!)(?!generate)(?!#select)(?!#each)(?!/each)(?!/select).*?\}\})", r"<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
                

        # format the generate command results
        display_out = re.sub(r"{{!--GMARKER_START_gen\$([^\$]*)\$--}}", start_generate_or_select, display_out)
        display_out = display_out.replace("{{!--GMARKER_END_gen$$--}}", "</span>")
        def click_loop_start(id, total_count, echo, color):
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
            alpha = 1.0 if not echo else 0.5
            out = f"</div><div style='display: none; opacity: {alpha}' id='{id}_{index}'>"
            return out
        display_out = re.sub(
            r"{{!--GMARKERmany_generate_start_([^_]+)_([0-9]+)\$([^\$]*)\$--}}",
            lambda x: click_loop_start(x.group(3), int(x.group(2)), x.group(1) == "True", "rgba(0, 165, 0, 0.25)"),
            display_out
        )
        display_out = re.sub(
            r"(?:--}})?{{!--GMARKERmany_generate_([^_]+)_([0-9]+)\$([^\$]*)\$--}}{{!--G ",
            lambda x: click_loop_mid(x.group(3), int(x.group(2)), x.group(1) == "True"),
            display_out
        )
        display_out = re.sub(r"--}}{{!--GMARKERmany_generate_end\$([^\$]*)\$--}}", "</div>", display_out)

        # format the each command results
        display_out = re.sub(r"{{!--GMARKER_START_each\$([^\$]*)\$--}}", start_each, display_out)
        display_out = re.sub(
            r"{{!--GMARKER_each_noecho_start_([^_]+)_([0-9]+)\$([^\$]*)\$--}}",
            lambda x: click_loop_start(x.group(3), int(x.group(2)), False, "rgb(100, 100, 100, 1)"),
            display_out
        )
        display_out = re.sub(
            r"{{!--GMARKER_each_noecho_([^_]+)_([0-9]+)\$([^\$]*)\$--}}",
            lambda x: click_loop_mid(x.group(3), int(x.group(2)), False),
            display_out
        )
        display_out = re.sub(r"{{!--GMARKER_each_noecho_end\$([^\$]*)\$--}}", "</div>", display_out)

        # format the geneach command results
        display_out = re.sub(r"{{!--GMARKER_START_geneach\$([^\$]*)\$--}}", start_each, display_out)
        
        # format the set command results
        display_out = re.sub(r"{{!--GMARKER_set\$([^\$]*)\$--}}", r"<div style='background-color: rgba(165, 165, 165, 0); border-radius: 4px 4px 4px 4px; border: 1px solid rgba(165, 165, 165, 1); border-left: 2px solid rgba(165, 165, 165, 1); border-right: 2px solid rgba(165, 165, 165, 1); padding-left: 0px; padding-right: 3px; color: rgb(165, 165, 165, 1.0); display: inline; font-weight: normal; overflow: hidden;'><div style='display: inline; background: rgba(165, 165, 165, 1); padding-right: 5px; padding-left: 4px; margin-right: 3px; color: #fff'>set</div>\1</div>", display_out)
        display_out = re.sub(r"{{!--GMARKER_START_set\$([^\$]*)\$--}}", lambda x: "<span style='display: inline;' title='{}'>".format(undo_html_encode(x.group(1))), display_out)

        display_out = re.sub(r"{{!--GMARKER_START_select\$([^\$]*)\$--}}", start_generate_or_select, display_out)
        display_out = display_out.replace("{{!--GMARKER_END_select$$--}}", "</span>")
        display_out = re.sub(r"{{!--GMARKER_START_variable_ref\$([^\$]*)\$--}}", lambda x: "<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='{}'>".format(undo_html_encode(x.group(1))), display_out)
        display_out = display_out.replace("{{!--GMARKER_END_variable_ref$$--}}", "</span>")
        display_out = display_out.replace("{{!--GMARKER_each$$--}}", "")#<div style='border-left: 1px dashed rgb(0, 0, 0, .2); border-top: 0px solid rgb(0, 0, 0, .2); margin-right: -4px; display: inline; width: 4px; height: 24px;'></div>")
        display_out = re.sub(r"{{!--GMARKER_START_block\$([^\$]*)\$--}}", start_block, display_out)
        display_out = re.sub(r"{{!--GMARKER_START_([^\$]*)\$([^\$]*)\$--}}", lambda x: "<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='{}'>".format(undo_html_encode(x.group(2))), display_out)
        display_out = re.sub(r"{{!--GMARKER_END_([^\$]*)\$\$--}}", "</span>", display_out)
        
        # display_out = re.sub(' and (?=.* and )', ', ', display_out)

        # strip out comments
        display_out = re.sub(r"{{~?!.*?}}", "", display_out)

        # re.sub(r"<div class='strip_leading_whitespace'")

        display_out = add_spaces(display_out)
        display_out = "<pre style='margin: 0px; padding: 0px; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'>"+display_out+"</pre>"

        return display_out

def add_spaces(s):
    """ This adds spaces so the browser will show leading and trailing newlines.
    """
    if s.startswith("\n"):
        s = " " + s
    if s.endswith("\n"):
        s = s + " "
    return s

_built_ins = {
    "gen": library.gen,
    "each": library.each,
    "geneach": library.geneach,
    "select": library.select,
    "if": library.if_,
    "unless": library.unless,
    "add": library.add,
    "subtract": library.subtract,
    "strip": library.strip,
    "block": library.block,
    "set": library.set,
    "await": library.await_,
    "role": library.role,
    "user": library.user,
    "system": library.system,
    "assistant": library.assistant,
    "break": library.break_,
    "equal": library.equal,
    "==": library.equal,
    "greater": library.greater,
    ">": library.greater,
    "less": library.less,
    "<": library.less,
    "contains": library.contains,
    "parse": library.parse
}

class DisplayThrottler():
    def __init__(self, display_function, throttle_limit):
        self.display_function = display_function
        self.throttle_limit = throttle_limit
        self._done = False
        self.last_time = 0
    
    async def run(self):
        self._data_event = asyncio.Event()
        self._done_event = asyncio.Event()
        while True:
            await self._data_event.wait()
            now = time.time()
            log.info("in DisplayThrottler run loop -- now: {}, last_time: {}, throttle_limit: {}".format(now, self.last_time, self.throttle_limit))
            if self._done or now - self.last_time >= self.throttle_limit:
                try:
                    self.display_function(last=self._done)
                except Exception as e:
                    self._done = True
                    raise e
                finally:
                    self.last_time = now
                    self._data_event.clear()
                    if self._done:
                        self._done_event.set()
                        break
            else:
                await asyncio.sleep(self.throttle_limit - (now - self.last_time))

    def __call__(self, last=False):
        if last:
            self._done = True
        self._data_event.set()

    async def done(self):
        return await self._done_event.wait()
