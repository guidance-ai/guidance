

# Harsha Experiment

from typing import Any
from IPython.display import clear_output, display, HTML
import html
import re
import threading
from functools import wraps
import queue
import time
from types import GeneratorType
import random
import asyncio
import janus
import copy
import types
from guidance._utils import InPlace


class LM:
    def __init__(self, model):
        self.model = model
        self._state = ""
        self._children = []
        self._event_queue = None
        self._event_parent = None
        self._silent = None
        self._inplace = None
        self._variables = {}
        self.endpoint = None

    def _html(self):
        display_out = html.escape(self._state)
        display_out = re.sub(r"&lt;\|\|_#NODISP_\|\|&gt;.*?&lt;\|\|_/NODISP_\|\|&gt;", "", display_out, flags=re.DOTALL)
        display_out = re.sub(r"&lt;\|\|_html:(.*?)_\|\|&gt;", lambda x: html.unescape(x.group(1)), display_out, flags=re.DOTALL)
        # display_out = re.sub(r"<|#color=([^\]+)|>.*?<|", "<span style=\"color: \1\">", display_out) # rgba(0, 165, 0, 0.25)
        display_out = "<pre style='margin: 0px; padding: 0px; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'>"+display_out+"</pre>"
        return display_out
    
    def _send_to_event_queue(self, value):
        if self._event_queue is not None:
            self._event_queue.put(value)
        if self._event_parent is not None:
            self._event_parent._send_to_event_queue(value)

    @property
    def silent(self):
        if self._silent is not None:
            return self._silent
        return False
    
    def _clone(self):
        new_lm = copy.copy(self)
        new_lm._event_queue = None
        if self._event_queue is not None:
            new_lm._event_parent = self
        new_lm._variables = self._variables.copy()
        new_lm._children = []
        self._children.append(new_lm)
        return new_lm
    
    def _inplace_append(self, value, force_silent=False):
        """This is used just internally."""
        self._state += str(value)
        if not self.silent and not force_silent:
            clear_output(wait=True)
            display(HTML(self._html()))
        self._send_to_event_queue(self)

    def _repr_html_(self):
        clear_output(wait=True)
        return self._html()
    
    def __str__(self) -> str:
        return re.sub(r"<\|\|_.*?_\|\|>", "", self._state)
    
    def __add__(self, value):
        assert not self._inplace
        new_lm = self._clone()
        new_lm._inplace_append(value)
        return new_lm
    
    def __iadd__(self, value):
        if not self._inplace:
            new_lm = self._clone()
        else:
            new_lm = self
        new_lm._inplace_append(value)
        return new_lm
    
    def __len__(self):
        return len(str(self))
    
    def __call__(self, s):
        return self + s
    
    def __setitem__(self, key, value):
        self._variables[key] = value

    def __getitem__(self, key):
        return self._variables[key]
    
    def _block_method(self, open_text, close_text, inner_text):
        new_lm = self._clone()

        if inner_text is not None:
            return self + open_text + inner_text + close_text
            new_lm._inplace_append(open_text)
            new_lm += inner_text
            new_lm._inplace_append(close_text)
            return new_lm
        else:
            def __enter__(self):
                return self + open_text
            
            def __exit__(self, exc_type, exc_value, traceback):
                self._rec_close(close_text)
            
            # bind the enter and exit methods
            new_lm._instance__enter__ = types.MethodType(__enter__, new_lm)
            new_lm._instance__exit__ = types.MethodType(__exit__, new_lm)

            return new_lm
        
    def __enter__(self):
        if hasattr(self, "_instance__enter__"):
            return self._instance__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, "_instance__exit__"):
            return self._instance__exit__(exc_type, exc_value, traceback)
        
    def _rec_close(self, close_string):
        for child in self._children:
            child._inplace_append(close_string, force_silent=len(child._children) > 1)
            child._rec_close(close_string)

    def __call__(self, s):
        return self + s
    
    def gen(lm, name=None, *, max_tokens=10, list_append=False, pattern=None, stop=None, stop_regex=None, n=1, temperature=0.0, top_p=1.0,
            logprobs=None, cache_seed=None, token_healing=None, stream=None, function_call="none", save_stop_text=False, **llm_kwargs):

        # set stream if we are interactive
        if stream is None and not lm.silent:
            stream = True

        new_lm = lm._clone()

        with InPlace(lm) as new_lm:
            new_lm += f"<||_html:<span style='background-color: rgba(0, 165, 0, 0.25)'>_||>"

            if name is not None:
                new_lm[name] = ""

            gen_obj = lm.session(
                str(lm), stop=stop, stop_regex=stop_regex, max_tokens=max_tokens, n=n, pattern=pattern,
                temperature=temperature, top_p=top_p, logprobs=logprobs, cache_seed=cache_seed, token_healing=token_healing,
                echo=getattr(lm, "logprobs", False), stream=stream, function_call=function_call, **llm_kwargs
            )

            if not isinstance(gen_obj, (types.GeneratorType, list, tuple)):
                gen_obj = [gen_obj]

            if n == 1:
                generated_value = ""
                logprobs_out = []
                if list_append:
                    lm[name] = lm.get(name, [])
                    lm[name].append("")
                    list_ind = len(lm[name])-1
                    if logprobs is not None:
                        lm[name+"_logprobs"] = lm.get(name+"_logprobs", [])
                        lm[name+"_logprobs"].append([])
                        assert len(len(lm[name])) == len(len(lm[name+"_logprobs"]))
                for resp in gen_obj:
                    new_text = resp["choices"][0].get("text", "")
                    generated_value += new_text
                    new_lm += new_text
                    if logprobs is not None:
                        logprobs_out.extend(resp["choices"][0]["logprobs"]["top_logprobs"])
                    if list_append:
                        lm[name][list_ind] = generated_value
                        if logprobs is not None:
                            lm[name+"_logprobs"][list_ind] = logprobs_out
                    elif name is not None:
                        lm[name] = generated_value
                        if logprobs is not None:
                            lm[name+"_logprobs"] = logprobs_out
                
                # save the final stopping text if requested
                if save_stop_text is not False:
                    if save_stop_text is True:
                        save_stop_text = name+"_stop_text"
                    lm[save_stop_text] = resp["choices"][0].get('stop_text', None)
                
                if list_append:
                    lm[name][list_ind] = generated_value
                elif name is not None:
                    lm[name] = generated_value

            # for c in rand_string:
            #     time.sleep(0.1) # simulate a long-running task
            #     new_lm += c
            #     if name is not None:
            #         new_lm[name] += c
            
            new_lm += f"<||_html:</span>_||>"
        
        return new_lm
    
    def select(lm, name=None, *, options):
        with InPlace(lm) as new_lm:
            new_lm += f"<||_html:<span style='background-color: rgba(0, 165, 0, 0.25)'>_||>"
            selected = random.choice(options)
            time.sleep(0.5) # simulate a long-running task
            new_lm += selected
            if name is not None:
                new_lm[name] = selected
            new_lm += f"<||_html:</span>_||>"
        return new_lm



