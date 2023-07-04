from typing import Any
from IPython.display import clear_output, display, HTML
import html
import re
import copy


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

    def __enter__(self):
        if hasattr(self, "instance__enter__"):
            return self.instance__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, "instance__exit__"):
            return self.instance__exit__(exc_type, exc_value, traceback)

    def __call__(self, s):
        return self + s


class ChatLM(LM):

    def get_role_start(self, role_name):
        return f"<|im_start|>{role_name}\n"
    
    def get_role_end(self, role_name):
        return "<|im_end|>"