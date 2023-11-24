try:
    from IPython.display import clear_output, display, HTML
except ImportError:
    clear_output = lambda wait=True: None
    display = lambda arg: None
    HTML = lambda arg: None
import html
import re
import copy
import time
import numpy as np
from .._grammar import StatelessFunction, string, _call_pool, _tag_pattern, Null, replace_model_variables, unreplace_model_variables, select

# define some constants we will reuse many times
_null_grammar = string('')
format_pattern = re.compile(r"<\|\|_.*?_\|\|>", flags=re.DOTALL)
nodisp_pattern = re.compile(r"&lt;\|\|_#NODISP_\|\|&gt;.*?&lt;\|\|_/NODISP_\|\|&gt;", flags=re.DOTALL)
html_pattern = re.compile(r"&lt;\|\|_html:(.*?)_\|\|&gt;", flags=re.DOTALL)

class Model:
    '''A guidance model object, which represents a sequence model in a given state.
    
    Model objects are immutable representations of model state, so whenever you change
    them you get a new model object. However, these copies share the "expensive"
    parts of the model like the the parameters and KV-cache, so making copies is cheap.
    '''

    open_blocks = {} # track what context blocks are open
    _grammar_only = 0 # a flag that tracks when we are forced to be executing only compiled grammars (like when we are inside a select)
    _throttle_refresh = 0 # a flag that tracks when we can throttle our display since we know future display calls are going to happen

    def __init__(self, echo=True):
        '''Build a new model state object.
        
        Parameters
        ----------
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        '''

        self.echo = echo
        self.token_count = 0 # tracks how many tokens our byte state represents
        self.max_display_rate = 0.2 # this controls how frequently we are allowed to redraw the display (in seconds)
        self.opened_blocks = {} # what context blocks have been opened but not closed

        # private attributes
        self._variables = {} # these are the state variables stored with the model
        self._cache_state = {} # mutable caching state used to save computation        
        self._state = "" # the current bytes that represent the state of the model
        self._event_queue = None # TODO: these are for streaming results in code, but that needs implemented
        self._event_parent = None
        self._last_display = 0 # used to track the last display call to enable throttling

    def __call__(self, grammar=None, max_tokens=100, n=1, top_p=1, temperature=0.0, ensure_bos_token=True):
        # TODO: turn this into "append" and make the models keep the grammar and parse as current state
        pass # meant to be overriden by subclasses

    @property
    def default_end_patterns(self):
        '''The default end patterns we should use for `gen` calls.
        TODO: move this logic into the gen call...we can do with if we allow model_variables to run functions.
        
        These patterns are computed dynamically by the model object because they can depend on
        what the current open roles are, which is something 
        '''

        # add the eos token
        parts = [self.eos_token]

        # add any active non-empty role ends. Ignore role ends that are spaces
        for role_end_str in self.opened_blocks.values():
            role_end_str = format_pattern.sub("", role_end_str)
            if len(role_end_str) > 0 and not re.fullmatch(r'\s+', role_end_str):
                parts.append(role_end_str)

        return select(parts)

    def _html(self):
        '''Generate HTML that displays the model object.'''
        display_out = self._state
        for context in reversed(self.opened_blocks):
            display_out += self.opened_blocks[context]
        display_out = html.escape(display_out)
        display_out = nodisp_pattern.sub("", display_out)
        display_out = html_pattern.sub(lambda x: html.unescape(x.group(1)), display_out)
        display_out = "<pre style='margin: 0px; padding: 0px; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'>"+display_out+"</pre>"
        return display_out
    
    def _send_to_event_queue(self, value):
        '''For streaming in code.
        
        TODO: Is this still needed?'''
        if self._event_queue is not None:
            self._event_queue.put(value)
        if self._event_parent is not None:
            self._event_parent._send_to_event_queue(value)
    
    def copy(self):
        '''Create a shallow copy of the model object.'''
        
        # start with a shallow copy
        new_lm = copy.copy(self)

        # then copy a few things we need deeper copies of
        new_lm._variables = self._variables.copy()
        new_lm.opened_blocks = self.opened_blocks.copy()
        
        # create a new clean event queue # TODO: can we delete this now?
        new_lm._event_queue = None
        if self._event_queue is not None:
            new_lm._event_parent = self
        
        return new_lm
    
    def _inplace_append(self, value, force_silent=False):
        '''This is the base way to add content to the current LM object that is being constructed.
        
        All updates to the model state should eventually use this function.
        Note this should only be used after making a copy, otherwise immutability would be violated.

        Parameters
        ----------
        value : bytes
            The bytes we should append to our current state.
        '''

        # update the byte state
        self._state += str(value) # TODO: make _state to be bytes not a string

        # see if we should update the display
        if self.echo and not force_silent:
            if Model._throttle_refresh > 0:
                curr_time = time.time()
                if curr_time - self._last_display < self.max_display_rate:
                    return # we are throttling the update
                else:
                    self._last_display = curr_time
        
            clear_output(wait=True)
            display(HTML(self._html()))
        
        # TODO: is this needed? This was for programmatic streaming...
        self._send_to_event_queue(self)
    
    def reset(self, clear_variables=True):
        '''This resets the state of the model object.
        
        Parameters
        ----------
        clear_variables : bool
            If we should clear all the model object's variables in addition to reseting the byte state.
        '''
        self._state = self._state[:0]
        if clear_variables:
            self._variables = {}
        return self

    def _repr_html_(self):
        clear_output(wait=True)
        return self._html()
    
    def _current_prompt(self):
        '''The current prompt in bytes (which is the state without the context close tags).'''
        return format_pattern.sub("", self._state)
    
    def __str__(self):
        '''A string representation of the current model object (that includes context closers).'''
        out = self._current_prompt()
        for context in reversed(self.opened_blocks):
            out += format_pattern.sub("", self.opened_blocks[context])
        return out
    
    def __add__(self, value):
        '''Adding is the primary mechanism for extending model state.
        
        Parameters
        ----------
        value : guidance grammar
            The grammar used to extend the current model.
        '''

        # create the new lm object we will return
        # (we need to do this since Model objects are immutable)
        lm = self.copy()

        # inside this context we are free to drop display calls that come too close together
        with throttle_refresh():

            # close any newly closed contexts
            for context in list(reversed(lm.opened_blocks)):
                if context not in Model.open_blocks and context in lm.opened_blocks:
                    close_text = lm.opened_blocks[context] # save so we can delete it before adding it
                    del lm.opened_blocks[context]
                    lm._inplace_append(close_text)

            # apply any newly opened contexts (new from this object's perspective)
            for context in Model.open_blocks:
                if context not in lm.opened_blocks:
                    lm.opened_blocks[context] = "" # mark this so we don't readd when computing the opener (even though we don't know the close text yet)
                    lm += context.opener
                    with grammar_only():
                        tmp = lm + context.closer
                    close_text = tmp._state[len(lm._state):] # get the new state added by calling the closer
                    lm.opened_blocks[context] = close_text
            
            # wrap raw string values
            if isinstance(value, str):
                is_id = False
                parts = re.split(_tag_pattern, value)
                
                # we have no embedded objects
                if len(parts) == 1:
                    lm._inplace_append(value)
                    out = lm
                
                # if we have embedded objects we have to convert the string to a grammar tree
                else:
                    partial_grammar = _null_grammar
                    lm.suffix = ""
                    for i,part in enumerate(parts):
                        if i < len(parts) - 1:
                            lm.suffix = parts[i+1]
                        if is_id:
                            call = _call_pool[part]
                            if isinstance(call, StatelessFunction):
                                partial_grammar += _call_pool[part]
                            else:
                                lm += partial_grammar
                                lm = _call_pool[part](lm)
                                partial_grammar = _null_grammar
                        elif part != "":
                            partial_grammar += string(part)
                        is_id = not is_id
                    out = lm + partial_grammar
            
            # if we find a null value we do nothing
            elif isinstance(value, Null):
                out = lm
            
            # run stateless functions (grammar nodes)
            elif isinstance(value, StatelessFunction):
                out = lm._run_stateless(value)
            
            # run stateful functions
            else:
                out = value(lm)
        
        # this flushes the display
        out._inplace_append("")

        return out
    
    def endswith(self, s):
        '''Checks if the current model state ends with the given value.'''
        return self._current_prompt().endswith(s)
    
    def __len__(self):
        '''The string length of the current state.
        
        TODO: This should change to the byte length...
        '''
        return len(str(self))
    
    def __setitem__(self, key, value):
        raise Exception("Model objects are immutable so you can't use __setitem__! Consider using the .set(key, value) method instead to create a new updated model object.")

    def __getitem__(self, key):
        return self._variables[key]
    
    def __contains__(self, item):
        return item in self._variables
    
    def get(self, key, default=None):
        '''Return the value of a variable, or a default value if the variable is not present.
        
        Parameters
        ----------
        key : str
            The name of the variable.
        default : any
            The value to return if the variable is not current set.
        '''
        return self._variables.get(key, default)

    def set(self, key, value):
        '''Return a new model with the given variable value set.
        
        Parameters
        ----------
        key : str
            The name of the variable to be set.
        value : any
            The value to set the variable to.
        '''
        copy = self.copy()
        copy._variables[key] = value
        return copy
    
    def remove(self, key):
        '''Return a new model with the given variable deleted.
        
        Parameters
        ----------
        key : str
            The variable name to remove.
        '''
        if key in self._variables:
            copy = self.copy()
            del copy._variables[key]
        else:
            copy = self
        return copy
    
    def get_cache(self):
        return self.engine.cache
    
    def tool_def(self, functions):

        self += """
# Tools

"""
        if len(functions) > 0:
            self += '''## functions

namespace functions {

'''
        for function in functions:
            self += f"""// {function['description']}
type {function['name']} = (_: {{"""
            for prop_name,prop_data in function["parameters"]["properties"].items():
                if "description" in prop_data:
                    self += f"\n// {prop_data['description']}\n"
                self += prop_name
                if prop_name not in function["parameters"]["required"]:
                    self += "?"
                self += ": "
                if "enum" in prop_data:
                    for enum in prop_data["enum"]:
                        self += f'"{enum}"'
                        if enum != prop_data["enum"][-1]:
                            self += " | "
                else:
                    self += prop_data["type"]
                
                if prop_name != list(function["parameters"]["properties"].keys())[-1]:
                    self += ",\n"
            self += """
}) => any;

"""
            self[function['name']] = function
        self += "} // namespace functions\n"
        
        return self

    def _run_stateless(lm, stateless_function, max_tokens=1000, temperature=0.0, top_p=1.0, n=1):
        assert Model._grammar_only == 0, "We can't run grammar parsing while in context free mode! (for example inside a block closer)"

        # This needs to be here for streaming
        # if name is not None:
        #     lm[name] = ""


        # replace ModelVariables with their actual values (note we save what we replaced so we can restore it later)
        replacements = replace_model_variables(stateless_function, lm)

        # start the generation stream
        gen_obj = lm(
            grammar=stateless_function, max_tokens=max_tokens, n=n,
            temperature=temperature, top_p=top_p
        )

        # single generation
        if n == 1:
            generated_value = ""
            # logprobs_out = []

            delayed_bytes = b""
            # last_is_generated = False
            for new_bytes, is_generated, new_bytes_log_prob, capture_groups, capture_group_log_probs, new_token_count in gen_obj:
                # convert the bytes to a string (delaying if we don't yet have a valid unicode string)
                lm.token_count += new_token_count
                new_bytes = delayed_bytes + new_bytes
                try:
                    new_text = new_bytes.decode("utf8")
                except UnicodeDecodeError:
                    delayed_bytes = new_bytes
                    continue
                delayed_bytes = b""

                if len(new_bytes) > 0:
                    generated_value += new_text
                    if is_generated:
                        lm += f"<||_html:<span style='background-color: rgba(0, 165, 0, {0.15 + 0.4 * (1 - np.exp(new_bytes_log_prob))}); border-radius: 3px;' title='{new_bytes_log_prob}'>_||>"
                    lm += new_text
                    if is_generated:
                        lm += "<||_html:</span>_||>"
                
                # last_is_generated = is_generated

                if len(capture_groups) > 0:
                    for k in capture_groups:
                        v = capture_groups[k]
                            
                        # see if we are in a list_append mode
                        if isinstance(v, list):
                            for inner_v in v:
                                # convert to a string if possible
                                # TODO: will need to not just always do this once we support images etc.
                                try:
                                    inner_v = inner_v.decode("utf8") if isinstance(inner_v, bytes) else inner_v
                                except UnicodeDecodeError:
                                    pass

                                if k not in lm or not isinstance(lm._variables[k], list):
                                    lm._variables[k] = []
                                lm._variables[k].append(inner_v)

                        # ...or standard assignment mode
                        else:
                            # convert to a string if possible
                            # TODO: will need to not just always do this once we support images etc.
                            try:
                                v = v.decode("utf8") if isinstance(v, bytes) else v
                            except UnicodeDecodeError:
                                pass
                            lm._variables[k] = v

            # if len(capture_groups) > 0:
            #     for k in capture_groups:
            #         v = capture_groups[k]
            #         lm[k] = v.decode("utf8") if isinstance(v, bytes) else v
        
        unreplace_model_variables(replacements)

        return lm

class Chat(Model):
    '''The base class for all chat-tuned models.'''
    
    def get_role_start(self, role_name, **kwargs):
        '''The starting grammar for a role.
        
        By default we follow the GPT role tag start conventions.
        
        Parameters
        ----------
        role_name : str
            The name of the role, like "user", or "assistant"
        kwargs : dict
            This kwargs are added to the role start as arguments.
        '''
        return "<|im_start|>"+role_name+"".join([f' {k}="{v}"' for k,v in kwargs.items()])+"\n"
    
    def get_role_end(self, role_name=None):
        '''The ending bytes for a role.
        
        Note that we cannot use a grammar in closers because they need to remain constant
        so we can append them whenever we need a representation before the final closing of the context.
        By default we follow the GPT role tag end conventions.
        
        Parameters
        ----------
        role_name : str
            The name of the role, like "user", or "assistant"
        '''
        return "<|im_end|>"
    
class Instruct(Model):
    '''The base class for all instruction-tuned models.'''

    def get_role_start(self, role_name, **kwargs):
        raise Exception("Subclasses need to define what the role start should be!")
    
    def get_role_end(self, role_name=None):
        raise Exception("Subclasses need to define what the role end should be!")
    
class GrammarOnly:
    def __enter__(self):
        Model._grammar_only += 1
    
    def __exit__(self, exc_type, exc_value, traceback):
        Model._grammar_only -= 1

def grammar_only():
    '''Returns a context manager that ensures only grammars are executed (not full python functions).'''
    return GrammarOnly()

class ThrottleRefresh:
    def __enter__(self):
        Model._throttle_refresh += 1
    
    def __exit__(self, exc_type, exc_value, traceback):
        Model._throttle_refresh -= 1

def throttle_refresh():
    '''Returns a context manager that allows the print statement to drop display calls above the throttle rate.'''
    return ThrottleRefresh()