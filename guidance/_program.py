import ast
import inspect
import re
import html
import uuid
import sys
import parsimonious
import warnings
import copy
import asyncio
import pathlib
import os
import traceback
import time
import datetime
from .llms import _openai
from . import _utils
import guidance

# load the javascript client code
file_path = pathlib.Path(__file__).parent.parent.absolute()
with open(file_path / "guidance" / "resources" / "main.js", encoding="utf-8") as f:
    js_data = f.read()

class Program:
    ''' A program template that can be compiled and executed to generate a new filled in (executed) program.

    Note that as the template gets executed {{!-- handlebars comment markers --}} get left in
    the generated output to mark where template tags used to be.
    '''

    def __init__(self, text, llm=None, cache_seed=0, logprobs=None, stream=False, echo=False, **kwargs):
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
        stream : bool (default False)
            If True, the program will be executed in "streaming" mode, where the program is executed
            asyncronously. If the program is displayed while streaming the output is generated per token.
        echo : bool (default False)
            If True, the program will be displayed as it is executed (this is an alternative to using stream=True
            and then displaying the async object).
        """

        # see if we were given a raw function instead of a string template
        # if so, convert it to a string template that calls the function
        if not isinstance(text, str):
            if callable(text):
                sig = inspect.signature(text)
                args = ""
                for name,_ in sig.parameters.items():
                    args += f" {name}={name}"
                fname = _utils.find_func_name(text, kwargs)
                kwargs[fname] = text
                text = "{{set (%s%s)}}" % (fname, args)
        
        # save the given parameters
        self._text = text
        self.llm = llm or guidance.llm
        self.cache_seed = cache_seed
        self.logprobs = logprobs
        self.stream = stream
        self.echo = echo
        
        # set our variables
        self.variables = {}
        self.variables.update(_built_ins)
        self.variables.update(kwargs)
        
        # set internal state variables
        self._id = str(uuid.uuid4())
        self.display_debounce_delay = 0.1 # the minimum time between display updates
        self._comm = None # front end communication object
        self._displaying_html = False # if we are displaying html (vs. text)
        self._executor = None # the ProgramExecutor object that is running the program
        self._last_display_update = 0 # the last time we updated the display (used for throttling updates)
        self._execute_complete = asyncio.Event() # fires when the program is done executing to resolve __await__
        self._displaying = echo # if we are displaying we need to update the display as we execute
        self._displayed = False # marks if we have been displayed in the client yet

        # see if we are in an ipython environment
        try:
            self._ipython = get_ipython()
        except:
            self._ipython = None

        # get or create an event loop
        if asyncio.get_event_loop().is_running():
            self.event_loop = asyncio.get_event_loop()
        else:
            self.event_loop = asyncio.new_event_loop()

        # find all the handlebars-style partial inclusion tags and replace them with the partial template
        def replace_partial(match):
            partial_name,args_string = match.group(1).split(" ", 1)
            if partial_name not in kwargs:
                raise ValueError("Partial '%s' not given in the keyword args:" % partial_name)
            out = "{{#block '"+partial_name+"'"
            if len(args_string) > 0:
                out += " " + args_string
            out += "}}" + kwargs[partial_name].text + "{{/block}}"
            self.variables = {**kwargs[partial_name].variables, **self.variables} # pull in the default vars from the partial
            return out
        self._text = re.sub(r"{{>(.*?)}}", replace_partial, self._text)
    
    def __repr__(self):
        return self.text
    
    def __getitem__(self, key):
        return self.variables[key]
    
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
        
        # mark that we are displaying (and so future execution updates should be displayed)
        self._displaying = True
        self._displaying_html = True
        
        # create the comm object if we don't have one
        if self._comm is None:
            self._comm = _utils.JupyterComm(self._id, self._ipython, self._interface_event)
        
        # build the html
        html = self._build_html(self.marked_text)
        html = f"""<div id="guidance-stop-button-{self._id}" style="cursor: pointer; margin: 0px; display: none; float: right; padding: 3px; border-radius: 4px 4px 4px 4px; border: 0px solid rgba(127, 127, 127, 1); padding-left: 10px; padding-right: 10px; font-size: 13px; background-color: rgba(127, 127, 127, 0.25);">Stop program</div><div id="guidance-content-{self._id}">{html}</div>
    <script type="text/javascript">{js_data}; window._guidanceDisplay("{self._id}");</script>"""
        
        # display the html
        from IPython.display import display
        display({"text/html": html}, display_id=self._id, raw=True, clear=True, include=["text/html"])

    async def _await_finish_execute(self):
        """ Used by self.__await__ to wait for the program to complete.
        """
        await self._execute_complete.wait() # wait for the program to finish executing
        return self

    def __await__(self):
        return self._await_finish_execute().__await__()
        
    def __call__(self, **kwargs):
        """ Execute this program with the given variable values and return a new executed/executing program.

        Note that the returned program might not be fully executed if `stream=True`. When streaming you need to
        use the python `await` keyword if you want to ensure the program is finished (note that is different than
        the `await` guidance langauge command, which will cause the program to stop execution at that point).
        """

        # create a new program object that we will execute in-place
        new_program = Program(
            self._text,
            self.llm,
            self.cache_seed,
            self.logprobs,
            self.stream,
            self.echo,

            # copy the (non-function) variables so that we don't modify the original program during execution
            # TODO: what about functions? should we copy them too?
            **{**{k: v if callable(v) else copy.deepcopy(v) for k,v in self.variables.items()}, **kwargs}
        )

        # create an executor for the new program (this also marks the program as executing)
        new_program._executor = ProgramExecutor(new_program)
        
        # if we are streaming schedule the program in the current event loop
        if self.stream:
            loop = asyncio.get_event_loop()
            assert self.event_loop.is_running()
            self.event_loop.create_task(new_program.execute())

        # if we are not streaming, we need to create a new event loop and run the program in it until it is done
        else:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(new_program.execute())
    
        return new_program
    
    def _update_display(self, last=False):
        """ Updates the display with the current marked text after debouncing.

        Parameters
        ----------
        last : bool
            If True, this is the last update and we should clear the send queue and prepare the
            UI for saving etc.
        """

        # this is always called during execution, and we only want to update the display if we are displaying
        if not self._displaying:
            return
        
        # debounce the display updates
        now = time.time()
        if last or (now - self._last_display_update > self.display_debounce_delay):
            if self._displaying_html:
                out = self._build_html(self.marked_text)
                
                # clear the send queue if this is the last update
                if last and self._comm:
                    self._comm.clear_send_queue()
                
                # send an update to the front end client if we have one...
                # TODO: we would like to call `display` for the last update so NB saving works, but see https://github.com/microsoft/vscode-jupyter/issues/13243 
                if self._comm and (not last or self._comm.is_open):
                    self._comm.send({"replace": out})
                    if last:
                        self._comm.send({"event": "complete"})
                
                # ...otherwise dump the client to the font end
                else:
                    from IPython.display import clear_output, display
                    if self._displayed:
                        clear_output(wait=False) # should use wait=True but that doesn't work in VSCode until after the April 2023 release
                    html = f"""<div id="guidance-stop-button-{self._id}" style="cursor: pointer; margin: 0px; display: none; float: right; padding: 3px; border-radius: 4px 4px 4px 4px; border: 0px solid rgba(127, 127, 127, 1); padding-left: 10px; padding-right: 10px; font-size: 13px; background-color: rgba(127, 127, 127, 0.25);">Stop program</div><div id="guidance-content-{self._id}">{out}</div>
                    <script type="text/javascript">{js_data}; window._guidanceDisplay("{self._id}");</script>"""
                    display({"text/html": html}, display_id=self._id, raw=True, clear=True, include=["text/html"])
                self._displayed = True
            
            self._last_display_update = time.time()

    async def execute(self):
        """ Execute the current program.

        Note that as execution progresses the program will be incrementally converted
        from a template into a completed string (with variables stored). At each point
        in this process the current template remains valid.
        """
        
        # if we are already displaying html, we need to yeild to the event loop so the jupyter comm can initialize
        if self._displaying_html:
            await asyncio.sleep(0)
        
        # run the program and capture the output
        await self._executor.run()
        self._text = self._executor.prefix

        # delete the executor and so mark the program as not executing
        self._executor = None

        # update the display with the final output
        self._update_display(last=True)

        # fire an event noting that execution is complete (this will release any await calls waiting on the program)
        self._execute_complete.set()
    
    def __getitem__(self, key):
        return self.variables[key]
    
    @property
    def text(self):
        # strip out the markers for the unformatted output
        return strip_markers(self.marked_text)
    
    @property
    def marked_text(self):
        if self._executor is not None:
            return self._executor.prefix
        else:
            return self._text
    
    def _build_html(self, text, last=False):
        output = text

        def start_generate_or_select(x):
            no_echo = "echo=False" in x.group(1)
            alpha = 1.0 if no_echo else 1.0
            
            # script that toggles the viisibility of the next element
            click_script = 'var e = this.nextElementSibling; if (e.style.display == "inline") { e.style.display = "none"; this.style.borderRight = "1px solid rgba(0, 165, 0, 0.25)"; } else { e.style.display = "inline"; this.style.borderRight = "0px";}'

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
        
        def start_block(x):
            escaped_tag = x.group(1)
            if "hidden=True" in escaped_tag:
                display = "none"
            else:
                display = "inline"
            return f"<span style='background-color: rgba(165, 165, 165, 0.1); display: {display};' title='{escaped_tag}'>"

        display_out = html.escape(output)
        display_out = re.sub(r"(\{\{generate.*?\}\})", r"<span style='background-color: rgba(0, 165, 0, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
        display_out = re.sub(r"(\{\{#select\{\{/select.*?\}\})", r"<span style='background-color: rgba(0, 165, 0, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
        display_out = re.sub(r"(\{\{#each [^'\"].*?\{\{/each.*?\}\})", r"<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
        display_out = re.sub(r"(\{\{(?!\!)(?!generate)(?!#select)(?!#each)(?!/each)(?!/select).*?\}\})", r"<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25);'>\1</span>", display_out, flags=re.DOTALL)
                

        # format the generate command results
        display_out = re.sub(r"{{!--GMARKER_START_generate\$([^\$]*)\$--}}", start_generate_or_select, display_out)
        display_out = display_out.replace("{{!--GMARKER_END_generate$$--}}", "</span>")
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
            r"{{!--GMARKER_generate_many_start_([^_]+)_([0-9]+)\$([^\$]*)\$--}}",
            lambda x: click_loop_start(x.group(3), int(x.group(2)), x.group(1) == "True", "rgba(0, 165, 0, 0.25)"),
            display_out
        )
        display_out = re.sub(
            r"{{!--GMARKER_generate_many_([^_]+)_([0-9]+)\$([^\$]*)\$--}}",
            lambda x: click_loop_mid(x.group(3), int(x.group(2)), x.group(1) == "True"),
            display_out
        )
        display_out = re.sub(r"{{!--GMARKER_generate_many_end\$([^\$]*)\$--}}", "</div>", display_out)

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
        
        # format the set command results
        display_out = re.sub(r"{{!--GMARKER_set\$([^\$]*)\$--}}", r"<div style='background-color: rgba(165, 165, 165, 0); border-radius: 4px 4px 4px 4px; border: 1px solid rgba(165, 165, 165, 1); border-left: 2px solid rgba(165, 165, 165, 1); border-right: 2px solid rgba(165, 165, 165, 1); padding-left: 0px; padding-right: 3px; color: rgb(165, 165, 165, 1.0); display: inline; font-weight: normal; overflow: hidden;'><div style='display: inline; background: rgba(165, 165, 165, 1); padding-right: 5px; padding-left: 4px; margin-right: 3px; color: #fff'>set</div>\1</div>", display_out)
        display_out = re.sub(r"{{!--GMARKER_START_set\$([^\$]*)\$--}}", r"<span style='display: inline;' title='\1'>", display_out)

        display_out = re.sub(r"{{!--GMARKER_START_select\$([^\$]*)\$--}}", start_generate_or_select, display_out)
        display_out = display_out.replace("{{!--GMARKER_END_select$$--}}", "</span>")
        display_out = re.sub(r"{{!--GMARKER_START_variable_ref\$([^\$]*)\$--}}", r"<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='\1'>", display_out)
        display_out = display_out.replace("{{!--GMARKER_END_variable_ref$$--}}", "</span>")
        display_out = display_out.replace("{{!--GMARKER_each$$--}}", "")#<div style='border-left: 1px dashed rgb(0, 0, 0, .2); border-top: 0px solid rgb(0, 0, 0, .2); margin-right: -4px; display: inline; width: 4px; height: 24px;'></div>")
        display_out = re.sub(r"{{!--GMARKER_START_block\$([^\$]*)\$--}}", start_block, display_out)
        display_out = re.sub(r"{{!--GMARKER_START_([^\$]*)\$([^\$]*)\$--}}", r"<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='\2'>", display_out)
        display_out = re.sub(r"{{!--GMARKER_END_([^\$]*)\$\$--}}", "</span>", display_out)
        
        # strip out comments
        display_out = re.sub(r"{{~?!.*?}}", "", display_out)

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

def strip_markers(s):
    return re.sub(r"{{!--G.*?--}}", r"", s, flags=re.MULTILINE | re.DOTALL)

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
block_command_call = command_name command_args
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
command_name = ~r"[a-z][a-z_0-9\.]*"i
variable_ref = !"or" !"else" ~r"[@a-z][a-z_0-9\.\[\]\"']*"i
variable_name = ~r"[@a-z][a-z_0-9]*"i
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

class StopCompletion(Exception):
    pass

async def _generate(variable_name="generated", partial_output=None, parse=False, stop=None, max_tokens=500, n=1, temperature=0.0, top_p=1.0, logprobs=None, hidden=False, parser_prefix=None, parser=None, prefix="", suffix="", next_text=None, prev_text=None, **kwargs):
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
    
    # set the cache seed to 0 if temperature is 0
    if temperature > 0:
        cache_seed = parser.program.cache_seed
        parser.program.cache_seed += 1
    else:
        cache_seed = 0

    # see if we should stream the results
    if n == 1:
        stream_generation = parser.program.stream is True or (parser.program.stream is None and parser.program.echo is True)
    else:
        stream_generation = False

    # call the LLM
    gen_obj = parser.program.llm(
        parser_prefix+prefix, stop=stop, max_tokens=max_tokens, n=n,
        temperature=temperature, top_p=top_p, logprobs=parser.program.logprobs, cache_seed=cache_seed,
        echo=parser.program.logprobs is not None, stream=stream_generation
    )

    if n == 1:
        generated_value = prefix
        partial_output(prefix)
        logprobs_out = []
        if not stream_generation:
            gen_obj = [gen_obj]
        for resp in gen_obj:
            await asyncio.sleep(0) # allow other tasks to run
            #log("parser.should_stop = " + str(parser.should_stop))
            if parser.should_stop:
                #log("Stopping generation")
                break
            generated_value += resp["choices"][0]["text"]
            partial_output(resp["choices"][0]["text"])
            if logprobs is not None:
                logprobs_out.extend(resp["choices"][0]["logprobs"])
            parser.set_variable(variable_name, generated_value)
            if logprobs is not None:
                parser.set_variable(variable_name+"_logprobs", logprobs_out)
        if hasattr(gen_obj, 'close'):
            gen_obj.close()
        generated_value += suffix
        partial_output(suffix)
        parser.set_variable(variable_name, generated_value)
        
        if parse:
            assert echo, "Cannot parse generated text if echo is disabled"
            subtree = grammar.parse(generated_value)
            return await parser.visit(subtree)
        else:
            # stop executing if we were interrupted
            if parser.should_stop:
                parser.executing = False
                parser.should_stop = False
            return
    else:
        assert not isinstance(gen_obj, list), "Streaming is only supported for n=1"
        generated_values = [prefix+choice["text"]+suffix for choice in gen_obj["choices"]]
        parser.set_variable(variable_name, generated_values)
        if logprobs is not None:
            parser.set_variable(variable_name+"_logprobs", [choice["logprobs"] for choice in gen_obj["choices"]])

        # TODO: we could enable the parsing to branch into multiple paths here, but for now we just complete the program with the first prefix
        generated_value = generated_values[0]

        # echoing with multiple completions is not standard behavior
        # this just uses the first generated value for completion and the rest as alternatives only used for the variable storage
        # we mostly support this so that the echo=False hiding behavior does not make multiple outputs more complicated than it needs to be in the UX
        # if echo:
        #     partial_output(generated_value) 
        
        id = uuid.uuid4().hex
        l = len(generated_values)
        out = "{{!--" + f"GMARKER_generate_many_start_{not hidden}_{l}${id}$" + "--}}"
        for i, value in enumerate(generated_values):
            if i > 0:
                out += "{{!--" + f"GMARKER_generate_many_{not hidden}_{i}${id}$" + "--}}"
            out += value
        partial_output(out + "{{!--" + f"GMARKER_generate_many_end${id}$" + "--}}")
        return
        # return "{{!--GMARKER_generate_many_start$$}}" + "{{!--GMARKER_generate_many$$}}".join([v for v in generated_values]) + "{{!--GMARKER_generate_many_end$$}}"
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

async def _each(list, block_content, parser, parser_prefix=None, parser_node=None, stop=None, hidden=False, filter=None, batch_generate=False, batch_generate_temperature=0.0, batch_generate_max_tokens=500, batch_generate_top_p=1.0):
    ''' Iterate over a list and execute a block for each item.
    '''
    assert len(block_content) == 1
    assert not (hidden and batch_generate), "Cannot use hidden=True and batch_generate together"
    assert not (hidden and isinstance(list, str)), "Cannot use hidden=True and variable length iteration together" # TODO: perhaps we can relax this?
    echo = not hidden

    # an empty string means the variable was not provided
    if list is None:
        list = []

    # apply a filter to the list if one was provided
    if filter is not None:

        # if the list is callable then we call it to get an iterable
        if callable(filter):
            # we support a set of optional arguments to the list function
            named_args = {}
            sig = inspect.signature(filter)
            if "template_context" in sig.parameters:
                named_args["template_context"] = {
                    "@block_text": block_content[0].text,
                    "@tokenizer": parser.program.tokenizer
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
    partial_out = ""
    
    # if the list is a string then it is the name of a variable to save a new list to
    if isinstance(list, str):
        # if stop is None:
        #     stop = "<|endoftext|>"
        # assert stop is not None, "Must provide a stop token when doing variable length iteration!"
        if stop is not None:
            stop_tokens = [parser.program.llm.encode(s) for s in stop]

        if not batch_generate:
            i = 0
            data = []
            while True:
                parser.variable_stack.append({})
                parser.variable_stack[-1]["@index"] = i
                parser.variable_stack[-1]["@first"] = i == 0
                parser.variable_stack[-1]["this"] = {}
                pos = len(parser.prefix)
                await parser.visit(block_content[0]) # fills out parser.prefix
                block_variables = parser.variable_stack.pop()["this"]
                data.append(block_variables)
                if not parser.executing:
                    block_text = parser.prefix[pos:]
                    block_text = block_text # make any unfinished this. references point to the last (unfinished) item
                    parser.prefix = parser.prefix[:pos] + parser.prefix[pos:].replace("this.", list+"[-1].")
                    break
                i += 1

                # see if we hit an await and so are not executing anymore
                if not parser.executing:
                    break

                # check if the block has thrown a stop iteration signal
                if parser.caught_stop_iteration:
                    parser.caught_stop_iteration = False
                    break

                # we run a quick generation to see if we have reached the end of the list (note the +2 tokens is to help be tolorant to whitespace)
                if stop is not None:
                    gen_obj = parser.program.llm(strip_markers(parser.prefix), stop=stop, max_tokens=len(stop_tokens)+2, temperature=0, cache_seed=0)
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
                cache_seed = parser.program.cache_seed
                parser.program.cache_seed += 1
            else:
                cache_seed = 0
            gen_obj = parser.program.llm(parser_prefix, stop=stop, max_tokens=batch_generate_max_tokens, temperature=batch_generate_temperature, top_p=batch_generate_top_p, cache_seed=cache_seed)
            generated_value = gen_obj["choices"][0]["text"]

            # parse the generated content (this assumes the generated content is syntactically correct)
            matches = re.finditer(pattern, generated_value)
            data = []
            for m in matches:#"{{!--" + f"GMARKER_START_{name}${node_text}$}}{out}{{!--GMARKER_END_{name}$$" + "}}"
                
                # get the variables that were generated
                match_dict = m.groupdict()
                if len(match_dict) == 1 and "this" in match_dict:
                    data.append(match_dict["this"])
                else:
                    data.append(match_dict)

                # recreate the output string with format markers added
                item_out = re.sub(
                    r"{{generate [\'\"]([^\'\"]+)[\'\"][^}]*}}",
                    lambda x: "{{!--GMARKER_START_generate$"+x.group()+"$--}}"+match_dict[x.group(1).replace("this.", "")]+"{{!--GMARKER_END_generate$$--}}",
                    block_content[0].text
                )
                out.append(item_out)
        
        # parser.get_variable(list, [])
        parser.set_variable(list, parser.get_variable(list, default_value=[]) + data)

    # if the list is not a string then it is a list of items to iterate over
    else:
        parser.variable_stack.append({})
        for i, item in enumerate(list):
            parser.variable_stack[-1]["@index"] = i
            parser.variable_stack[-1]["@first"] = i == 0
            parser.variable_stack[-1]["@last"] = i == len(list) - 1
            parser.variable_stack[-1]["this"] = item
            item_out = await parser.visit(block_content[0])
            if not echo:
                parser._trim_prefix(item_out)
            out.append(item_out)
        parser.variable_stack.pop()
    
    suffix = ""
    if not parser.executing:
        if isinstance(list, str):
            suffix = partial_out + parser_node.text
    
    if echo:
        return "{{!--GMARKER_each$$--}}" + "{{!--GMARKER_each$$--}}".join(out) + "{{!--GMARKER_each$$--}}" + suffix
    else:
        id = uuid.uuid4().hex
        l = len(out)
        out_str = prefix + "{{!--" + f"GMARKER_each_noecho_start_{echo}_{l}${id}$" + "--}}"
        for i, value in enumerate(out):
            if i > 0:
                out_str += "{{!--" + f"GMARKER_each_noecho_{echo}_{i}${id}$" + "--}}"
            out_str += value
        return out_str + "{{!--" + f"GMARKER_each_noecho_end${id}$" + "--}}"

        # return "{{!--GMARKER_each_noecho$$}}" + "{{!--GMARKER_each_noecho$$}}".join(out) + "{{!--GMARKER_each_noecho$$}}"

def _select(variable_name="selected", block_content=None, parser=None, partial_output=None, parser_prefix=None, logprobs=None):
    ''' Select a value from a list of choices.
    '''
    assert len(block_content) > 1
    options = [block_content[0].text]
    for i in range(1, len(block_content), 2):
        assert block_content[i].text == "{{or}}"
        options.append(block_content[i+1].text)

    option_tokens = [parser.program.llm.encode(option) for option in options]

    option_logprobs = {}
    for option in option_tokens:
        option_logprobs[parser.program.llm.decode(option)] = 0

    # [TODO] we should force the LM to generate a valid specific option
    #        for openai this means setting logprobs to valid token ids
    gen_obj = parser.program.llm(
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
                option_string = parser.program.llm.decode(option)
                option_logprobs[option_string] += top_logprobs[i].get(parser.program.llm.decode([option[i]]), -100)
    
    # penalize options that are too long
    for option in option_tokens:
        if len(option) > len(top_logprobs):
            option_logprobs[parser.program.llm.decode(option)] -= 100

    selected_option = max(option_logprobs, key=option_logprobs.get)
    parser.set_variable(variable_name, selected_option)
    if logprobs is not None:
        parser.set_variable(logprobs, option_logprobs)
    
    if max(option_logprobs.values()) <= -100:
        raise ValueError("No valid option generated in #select, this could be fixed if we used a tokenizer and forced the LM to use a valid option! The top logprobs were" + str(top_logprobs))
    
    partial_output(selected_option)

    return selected_option

async def _if(value, block_content, parser, reverse=False):
    ''' Standard if/else statement.
    '''
    assert len(block_content) in [1,3] # we don't support elseif yet...
    options = [block_content[0]]
    for i in range(1, len(block_content), 2):
        assert block_content[i].text == "{{else}}"
        options.append(block_content[i+1])

    if isinstance(value, str):
        value2 = value
        value = value.lower().strip() in ["true", "yes", "on", "t", "y", "ok", "okay"]
    
    if reverse:
        value = not value
    
    if value:
        return await parser.visit(options[0])
    elif len(options) > 1:
        return await parser.visit(options[1])
    else:
        return ""

async def _unless(value, block_content, parser):
    return await _if(value, block_content, parser, reverse=True)

async def _block(name=None, block_content=None, parser=None, hidden=False):
    ''' Generic block definition.
    '''
    assert parser is not None
    
    pos = len(parser.prefix)
    out = await parser.visit(block_content[0])
    if name is not None:
        parser.set_variable(name, strip_markers(out))
    if hidden:
        new_content = parser.prefix[pos:]
        parser.reset_prefix(pos)
        parser.extend_prefix("{{!--GHIDDEN:"+new_content.replace("--}}", "--_END_END")+"--}}")
    
    return out

def _set(name, value=None, hidden=None, parser=None):
    ''' Set the value of a variable or set of variables.

    Parameters
    ----------
    name : str or dict
        If a string, the name of the variable to set. If a dict, the keys are the variable names and the values are the values to set.
    value : str, optional
        The value to set the variable to. Only used if `name` is a string.
    hidden : bool, optional
        If True, the variable will be set but not printed in the output.
    '''
    assert parser is not None

    if not parser.executing:
        return ""

    if isinstance(name, dict):
        assert hidden is not False, "hidden cannot be False if setting multiple variables!"
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
        return "{{!--GMARKER_set$" + out + "$--}}"
    else:
        parser.set_variable(name, value)
        if hidden is not True:
            return value
        else:
            out = "{{set "+name+"=" + str(value) + "}}"
            return "{{!--GMARKER_set$" + out + "$--}}"

async def _await(name, parser=None):
    ''' Awaits a value by stopping execution if the value does not yet exist.
    '''

    # stop the program completion if we are waiting for a value to be set
    # this will result in a partially completed program that we can then finish
    # later (by calling it again with the variable we need)
    if name not in parser.program.variables:
        parser.executing = False
    else:
        value = parser.program.variables[name]
        del parser.program.variables[name]
        return value
    
    # cache = parser.program._await_cache
    # while name not in cache:
    #     parser.program.finish_execute() # allow the program to finish the current call (since we're waiting for a value from the next call now)
    #     # TODO: instead of waiting here, we should just single we are stopping the program completion here
    #     #       and then let all the containing elements record their state into a new program string that
    #     #       we can then use to continue the program completion later in a new object.
    #     cache.update(await parser.program._await_queue.get())
    #     pass
    # value = cache[name]
    # del cache[name]
    # return value


_built_ins = {
    "generate": _generate,
    "each": _each,
    "select": _select,
    "if": _if,
    "unless": _unless,
    "add": _add,
    "subtract": _subtract,
    "strip": _strip,
    "block": _block,
    "set": _set,
    "await": _await
}